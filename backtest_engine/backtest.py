"""QuantAI Unified — backtest.py
Engine de backtest com:
- Rebalanceamento mensal/trimestral + condicional (desvio > threshold)
- Custos de transação (bps) + slippage (bps) separados
- Métricas completas: CAGR, Sharpe, Sortino, MDD, Calmar, VaR, Info Ratio
- Comparação com benchmark ^BVSP
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Resultados completos do backtest."""
    equity_curve: pd.DataFrame       # strategy + benchmark (diário)
    weights: pd.DataFrame            # Pesos por data de rebalanceamento
    positions: pd.DataFrame          # Pesos diários interpolados
    trades: pd.DataFrame             # Histórico de trades (∆pesos)
    metrics: Dict                    # KPIs da estratégia
    benchmark_metrics: Dict          # KPIs do benchmark
    rebalance_dates: List[pd.Timestamp]
    regime_history: Dict[str, str]   # data → regime


def run_backtest(prices: pd.DataFrame, config: Dict) -> BacktestResult:
    """
    Executa backtest completo da estratégia quantitativa unificada.
    """
    from models.strategy import compute_factor_scores
    from portfolio.optimizer import optimize_portfolio, compute_sector_exposure
    from models.ml_models import compute_expected_returns_ml

    bt   = config.get("backtest", {})
    sel  = config.get("selection", {})
    benchmark = bt.get("benchmark", "^BVSP")

    start_dt = pd.to_datetime(bt.get("start_date", "2021-01-01"))
    end_dt   = pd.to_datetime(bt.get("end_date",   "2024-12-31"))
    capital  = float(bt.get("initial_capital", 100_000))
    tc_bps   = float(bt.get("transaction_cost_bps", 10))
    slip_bps = float(bt.get("slippage_bps", 5))
    total_cost_rate = (tc_bps + slip_bps) / 10_000

    rebal_freq   = bt.get("rebalance_frequency", "monthly")
    rebal_thresh = float(sel.get("rebalance_threshold", 0.05))
    sector_map   = config.get("sector_map", {})

    # Filtra período
    prices_bt = prices.loc[start_dt:end_dt].copy()
    assets = [c for c in prices.columns if c != benchmark]

    # Datas de rebalanceamento programadas
    if rebal_freq == "monthly":
        rebal_dates = _monthly_rebal_dates(prices_bt.index)
    else:
        rebal_dates = _quarterly_rebal_dates(prices_bt.index)

    logger.info(
        "Iniciando backtest: %s → %s | %d rebalanceamentos programados",
        start_dt.date(), end_dt.date(), len(rebal_dates),
    )

    # Estado inicial
    current_weights = pd.Series(dtype=float)
    weights_history: Dict[pd.Timestamp, pd.Series] = {}
    trades_history: List[Dict] = []
    regime_history: Dict[str, str] = {}
    equity = capital
    equity_series: Dict[pd.Timestamp, float] = {}
    bench_series: Dict[pd.Timestamp, float] = {}

    bench_start = prices_bt[benchmark].dropna().iloc[0] if benchmark in prices_bt.columns else None
    bench_capital = capital

    prev_close: Optional[pd.Series] = None
    actual_rebal_dates: List[pd.Timestamp] = []

    for date, row in prices_bt.iterrows():
        if prev_close is None:
            prev_close = row
            equity_series[date] = capital
            bench_series[date] = capital
            continue

        # Atualiza equity com retornos dos pesos correntes
        if not current_weights.empty:
            daily_ret = 0.0
            for asset, w in current_weights.items():
                if asset in row.index and asset in prev_close.index:
                    if prev_close[asset] > 0:
                        daily_ret += w * (row[asset] / prev_close[asset] - 1)
            equity *= (1 + daily_ret)

        # Atualiza benchmark
        if benchmark in row.index and benchmark in prev_close.index and prev_close[benchmark] > 0:
            bench_ret = row[benchmark] / prev_close[benchmark] - 1
            bench_capital *= (1 + bench_ret)

        equity_series[date] = equity
        bench_series[date] = bench_capital

        # Verifica se deve rebalancear (programado OU desvio excessivo)
        is_rebal_date = date in rebal_dates
        has_drift = _check_drift(current_weights, row, prev_close, rebal_thresh)
        should_rebal = is_rebal_date or has_drift

        if should_rebal:
            # Calcula scores de fator
            factor_result = compute_factor_scores(prices_bt, date, config, sector_map)

            # Expected returns via ML
            try:
                exp_returns = compute_expected_returns_ml(
                    prices_bt, factor_result.selected, date, config
                )
            except Exception:
                exp_returns = None

            # Otimiza portfólio
            new_weights = optimize_portfolio(
                prices_bt.loc[:date],
                factor_result.selected,
                config,
                exp_returns,
            )

            if new_weights.empty:
                new_weights = current_weights

            # Calcula custo de transação (turnover × custo)
            turnover = _compute_turnover(current_weights, new_weights)
            cost = turnover * total_cost_rate
            equity *= (1 - cost)

            # Registra trade
            trades_history.append({
                "date": date,
                "turnover": turnover,
                "cost_pct": cost * 100,
                "regime": factor_result.regime,
                "triggered_by": "schedule" if is_rebal_date else "drift",
            })

            weights_history[date] = new_weights
            regime_history[str(date.date())] = factor_result.regime
            current_weights = new_weights
            actual_rebal_dates.append(date)

            logger.debug(
                "[%s] Rebalanceamento | Turnover: %.1f%% | Custo: %.3f%% | Regime: %s",
                date.strftime("%Y-%m-%d"), turnover * 100, cost * 100, factor_result.regime
            )

        prev_close = row

    # Monta DataFrames de resultado
    equity_df = pd.DataFrame({
        "strategy":  pd.Series(equity_series),
        "benchmark": pd.Series(bench_series),
    })

    weights_df = pd.DataFrame(weights_history).T.fillna(0)
    positions_df = weights_df.reindex(prices_bt.index).ffill().fillna(0)
    trades_df = pd.DataFrame(trades_history)

    # Calcula métricas
    strat_metrics = _compute_metrics(equity_df["strategy"], capital, tc_bps + slip_bps)
    bench_metrics  = _compute_metrics(equity_df["benchmark"], capital, 0)

    logger.info(
        "Backtest concluído | CAGR: %.1f%% | Sharpe: %.2f | MDD: %.1f%%",
        strat_metrics["cagr"] * 100,
        strat_metrics["sharpe"],
        strat_metrics["max_drawdown"] * 100,
    )

    return BacktestResult(
        equity_curve=equity_df,
        weights=weights_df,
        positions=positions_df,
        trades=trades_df,
        metrics=strat_metrics,
        benchmark_metrics=bench_metrics,
        rebalance_dates=actual_rebal_dates,
        regime_history=regime_history,
    )


# ---------------------------------------------------------------------------
# Métricas de performance (Seção 5 do relatório)
# ---------------------------------------------------------------------------

def _compute_metrics(equity: pd.Series, initial_capital: float, cost_bps: float) -> Dict:
    """Computa todas as métricas de performance da Seção 5 do relatório."""
    equity = equity.dropna()
    if equity.empty or equity.iloc[0] == 0:
        return {}

    returns = equity.pct_change().dropna()
    n_years = len(equity) / 252
    rf_annual = 0.105   # Selic ~10.5% a.a. (benchmark renda fixa BR)
    rf_daily = rf_annual / 252

    # CAGR
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / max(n_years, 0.01)) - 1

    # Retorno total
    total_return = equity.iloc[-1] / equity.iloc[0] - 1

    # Volatilidade anualizada
    vol = returns.std() * np.sqrt(252)

    # Sharpe Ratio (com Selic como Rf)
    excess = returns - rf_daily
    sharpe = (excess.mean() * 252) / (returns.std() * np.sqrt(252)) if vol > 0 else 0

    # Sortino Ratio (apenas volatilidade negativa)
    neg_returns = returns[returns < 0]
    downside_vol = neg_returns.std() * np.sqrt(252) if len(neg_returns) > 0 else 1e-9
    sortino = (cagr - rf_annual) / downside_vol if downside_vol > 0 else 0

    # Maximum Drawdown
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Drawdown duration (dias)
    in_dd = drawdown < 0
    dd_duration = 0
    max_dd_dur = 0
    for v in in_dd:
        if v:
            dd_duration += 1
            max_dd_dur = max(max_dd_dur, dd_duration)
        else:
            dd_duration = 0

    # Calmar Ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # VaR 95% (1 dia)
    var_95 = float(returns.quantile(0.05))

    # Win Rate
    win_rate = float((returns > 0).mean())

    # Hit Rate mensal vs benchmark (será calculado externamente com benchmark)
    monthly_ret = equity.resample("ME").last().pct_change().dropna()
    positive_months = (monthly_ret > 0).sum()
    hit_rate = positive_months / len(monthly_ret) if len(monthly_ret) > 0 else 0

    # Turnover médio mensal (a ser preenchido externamente)
    return {
        "cagr": float(cagr),
        "total_return": float(total_return),
        "volatility": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "max_drawdown_duration_days": int(max_dd_dur),
        "calmar": float(calmar),
        "var_95_daily": float(var_95),
        "win_rate": float(win_rate),
        "monthly_hit_rate": float(hit_rate),
        "final_equity": float(equity.iloc[-1]),
        "n_years": float(n_years),
        "cost_bps": cost_bps,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monthly_rebal_dates(index: pd.DatetimeIndex) -> set:
    """Última data útil de cada mês."""
    return set(index.to_frame().groupby([index.year, index.month]).last().iloc[:, 0])


def _quarterly_rebal_dates(index: pd.DatetimeIndex) -> set:
    """Última data útil de cada trimestre."""
    return set(index.to_frame().groupby([index.year, index.quarter]).last().iloc[:, 0])


def _compute_turnover(old_w: pd.Series, new_w: pd.Series) -> float:
    """Turnover = soma dos |∆pesos| / 2."""
    all_assets = set(old_w.index) | set(new_w.index)
    total = sum(
        abs(new_w.get(a, 0) - old_w.get(a, 0))
        for a in all_assets
    )
    return total / 2


def _check_drift(
    current_w: pd.Series,
    current_prices: pd.Series,
    prev_prices: pd.Series,
    threshold: float,
) -> bool:
    """Verifica se os pesos da carteira desviaram mais que threshold dos alvos."""
    if current_w.empty:
        return False
    drifted = {}
    for asset, target in current_w.items():
        if asset in current_prices.index and asset in prev_prices.index:
            if prev_prices[asset] > 0:
                drift = abs(current_prices[asset] / prev_prices[asset] - 1) * target
                drifted[asset] = drift
    total_drift = sum(drifted.values())
    return total_drift > threshold