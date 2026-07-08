"""QuantAI Unified — strategy.py
Motor quantitativo multi-fator com:
- Momentum 3M/6M/12M (estratégia 12-1 clássica)
- Qualidade de tendência (R²)
- Volatilidade realizada
- RSI(14), ATR(14), Bollinger Bands
- Filtro de regime (SMA200)
- Neutralização por setor (melhoria original)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FactorScores:
    """Scores de fator por ativo em uma data de rebalanceamento."""
    scores: pd.Series          # Score composto final
    factor_breakdown: pd.DataFrame   # Score por fator
    regime: str                # "bull" | "lateral" | "bear"
    selected: List[str]        # Ativos selecionados (top_n)


def compute_factor_scores(
    prices: pd.DataFrame,
    rebal_date: pd.Timestamp,
    config: Dict,
    sector_map: Optional[Dict[str, str]] = None,
    macro_data: Optional[Dict] = None,
) -> FactorScores:
    """
    Calcula scores multi-fator para todos os ativos em uma data de rebalanceamento.
    Aplica neutralização setorial e filtro de regime.
    """
    sel = config.get("selection", {})
    fct = config.get("factors", {})
    benchmark = config.get("backtest", {}).get("benchmark", "^BVSP")

    # Janela de lookback: 252 dias úteis
    lookback_start = rebal_date - pd.offsets.BDay(280)
    window = prices.loc[lookback_start:rebal_date].copy()

    assets = [c for c in prices.columns if c != benchmark]
    available = [a for a in assets if a in window.columns and window[a].dropna().shape[0] >= 60]

    if not available:
        logger.warning("Nenhum ativo disponível em %s", rebal_date)
        return FactorScores(
            scores=pd.Series(dtype=float),
            factor_breakdown=pd.DataFrame(),
            regime="unknown",
            selected=[],
        )

    factor_weights = fct.get("momentum_weights", {"mom_3m": 0.25, "mom_6m": 0.35, "mom_12m": 0.40})
    trend_w  = fct.get("trend_quality_weight", 0.20)
    vol_w    = fct.get("volatility_weight", 0.20)
    rsi_w    = fct.get("rsi_weight", 0.15)
    macro_w  = fct.get("macro_weight", 0.05)

    # === Regime de mercado (SMA200 no benchmark) ===
    regime = _detect_regime(window, benchmark)

    # === Calcula indicadores por ativo ===
    rows = []
    for asset in available:
        s = window[asset].dropna()
        if len(s) < 60:
            continue
        row = {"asset": asset}
        row.update(_compute_momentum(s))
        row.update(_compute_trend_quality(s))
        row.update(_compute_volatility(s))
        row.update(_compute_rsi(s))
        row.update(_compute_bollinger(s))
        row.update(_compute_atr(s))
        rows.append(row)

    if not rows:
        return FactorScores(pd.Series(dtype=float), pd.DataFrame(), regime, [])

    df = pd.DataFrame(rows).set_index("asset")

    # === Normalização Z-score geral ===
    df_zscore = df.apply(lambda col: _zscore(col), axis=0)

    # === Neutralização setorial (melhoria original) ===
    if sector_map:
        df_zscore = _sector_neutralize(df_zscore, sector_map)

    # === Score composto ===
    mom_cols = ["mom_3m", "mom_6m", "mom_12m"]
    mom_w_values = [factor_weights.get(c, 1/3) for c in mom_cols]
    mom_w_norm = np.array(mom_w_values) / sum(mom_w_values)

    score = pd.Series(0.0, index=df_zscore.index)

    # Momentum ponderado (excluindo último mês — estratégia 12-1)
    for col, w in zip(mom_cols, mom_w_norm):
        if col in df_zscore.columns:
            mom_contrib = (1.0 - fct.get("momentum_weights", {}).get("mom_3m", 0.25))
            score += w * df_zscore[col] * mom_contrib

    if "trend_r2" in df_zscore.columns:
        score += trend_w * df_zscore["trend_r2"]
    if "vol_21d" in df_zscore.columns:
        score += vol_w * (-df_zscore["vol_21d"])   # Menor vol → melhor score
    if "rsi" in df_zscore.columns:
        # RSI: penaliza sobrecomprados (>70) e sobrevendidos (<30)
        rsi_score = -abs(df_zscore["rsi"])
        score += rsi_w * rsi_score

    # Ajuste de regime
    if regime == "bear":
        score *= 0.5   # Reduz exposição em mercados de baixa

    # === Seleciona top_n ===
    top_n = sel.get("top_n", 8)
    selected = score.nlargest(top_n).index.tolist()

    breakdown_cols = [c for c in ["mom_3m", "mom_6m", "mom_12m", "trend_r2", "vol_21d", "rsi", "bb_position"] if c in df_zscore.columns]
    factor_breakdown = df_zscore[breakdown_cols].copy()
    factor_breakdown["score_final"] = score

    logger.info(
        "[%s] Regime: %s | Top %d: %s",
        rebal_date.strftime("%Y-%m"),
        regime,
        top_n,
        ", ".join(selected[:5]),
    )

    return FactorScores(
        scores=score,
        factor_breakdown=factor_breakdown,
        regime=regime,
        selected=selected,
    )


# ---------------------------------------------------------------------------
# Indicadores técnicos
# ---------------------------------------------------------------------------

def _compute_momentum(s: pd.Series) -> Dict:
    """Momentum 3M (63d), 6M (126d), 12M (252d) — estratégia 12-1 (exclui último mês)."""
    n = len(s)
    ret = {}
    # Exclui o último mês (21 dias) para evitar reversão de curto prazo
    skip = 21
    for label, days in [("mom_3m", 63), ("mom_6m", 126), ("mom_12m", 252)]:
        if n >= days + skip:
            ret[label] = (s.iloc[-(skip + 1)] / s.iloc[-(days + skip)] - 1)
        elif n >= days:
            ret[label] = (s.iloc[-1] / s.iloc[-days] - 1)
        else:
            ret[label] = 0.0
    return ret


def _compute_trend_quality(s: pd.Series, window: int = 60) -> Dict:
    """R² da regressão linear dos preços nos últimos 60 dias."""
    try:
        from sklearn.linear_model import LinearRegression
        y = s.iloc[-window:].values
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        ss_res = np.sum((y - model.predict(x)) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        slope_sign = 1 if model.coef_[0] > 0 else -1
        return {"trend_r2": r2 * slope_sign}
    except Exception:
        return {"trend_r2": 0.0}


def _compute_volatility(s: pd.Series) -> Dict:
    """Volatilidade anualizada em janelas de 21d e 63d."""
    rets = s.pct_change().dropna()
    vol_21 = rets.iloc[-21:].std() * np.sqrt(252) if len(rets) >= 21 else np.nan
    vol_63 = rets.iloc[-63:].std() * np.sqrt(252) if len(rets) >= 63 else np.nan
    return {"vol_21d": vol_21, "vol_63d": vol_63}


def _compute_rsi(s: pd.Series, period: int = 14) -> Dict:
    """RSI(14) — normalizado de 0 a 1 → pontuação neutra em 0.5."""
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return {"rsi": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0}


def _compute_bollinger(s: pd.Series, window: int = 20) -> Dict:
    """Posição relativa dentro das Bollinger Bands (0=banda inferior, 1=superior)."""
    if len(s) < window:
        return {"bb_position": 0.5}
    sma   = s.rolling(window).mean().iloc[-1]
    std   = s.rolling(window).std().iloc[-1]
    upper = sma + 2 * std
    lower = sma - 2 * std
    pos   = (s.iloc[-1] - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
    return {"bb_position": float(np.clip(pos, 0, 1))}


def _compute_atr(s: pd.Series, period: int = 14) -> Dict:
    """ATR(14) normalizado pelo preço."""
    if len(s) < period + 1:
        return {"atr_pct": np.nan}
    highs  = s.rolling(2).max()
    lows   = s.rolling(2).min()
    tr     = highs - lows
    atr    = tr.rolling(period).mean().iloc[-1]
    atr_pct = atr / s.iloc[-1] if s.iloc[-1] > 0 else np.nan
    return {"atr_pct": float(atr_pct)}


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _zscore(series: pd.Series) -> pd.Series:
    """Z-score robusto usando median/IQR para resistência a outliers."""
    median = series.median()
    iqr    = series.quantile(0.75) - series.quantile(0.25)
    if iqr == 0:
        return (series - median).fillna(0)
    return ((series - median) / iqr).fillna(0).clip(-3, 3)


def _sector_neutralize(
    df_zscore: pd.DataFrame,
    sector_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Neutralização setorial: remove o viés de setor normalizando os z-scores
    dentro de cada grupo setorial. Evita que setores 'quentes' dominem.
    """
    df_out = df_zscore.copy()
    sectors = pd.Series(sector_map).reindex(df_zscore.index)
    for sector, group in df_zscore.groupby(sectors):
        if len(group) < 2:
            continue
        for col in df_zscore.columns:
            if col in group.columns:
                g_vals = group[col]
                df_out.loc[group.index, col] = _zscore(g_vals)
    return df_out


def _detect_regime(window: pd.DataFrame, benchmark: str) -> str:
    """
    Detecta regime de mercado via SMA200 no benchmark.
    bull: preço > SMA200 | bear: preço < SMA200 * 0.95 | lateral: entre
    """
    if benchmark not in window.columns:
        return "bull"
    bvsp = window[benchmark].dropna()
    if len(bvsp) < 200:
        return "bull"
    sma200 = bvsp.rolling(200).mean().iloc[-1]
    price  = bvsp.iloc[-1]
    if price > sma200:
        return "bull"
    elif price < sma200 * 0.95:
        return "bear"
    return "lateral"