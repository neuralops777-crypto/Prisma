"""QuantAI Unified — data.py
Coleta de dados de mercado (yfinance) e macro (BCB/SGS).
Modo offline com dados sintéticos realistas quando sem internet.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dados de mercado (yfinance)
# ---------------------------------------------------------------------------

def load_prices(config, force_download: bool = False) -> pd.DataFrame:
    """
    Carrega preços ajustados de fechamento para todos os ativos + benchmark.
    Usa cache Parquet (6h TTL). Fallback para dados sintéticos offline.
    """
    from configs.config import QuantAIConfig
    cfg: QuantAIConfig = config

    cache_path = cfg.data_dir / "prices.parquet"
    assets_with_bench = list(cfg.assets) + [cfg.benchmark]

    # Verifica cache (TTL 6 horas)
    if not force_download and cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 6:
            logger.info("Carregando preços do cache Parquet (%.1fh atrás)", age_hours)
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning("Falha ao ler cache: %s", e)

    try:
        import yfinance as yf
        logger.info("Baixando dados do Yahoo Finance (%d ativos)...", len(assets_with_bench))
        raw = yf.download(
            assets_with_bench,
            start=cfg.start_date,
            end=cfg.end_date,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].copy()
        else:
            prices = raw[["Close"]].copy()
            prices.columns = assets_with_bench[:1]

        prices = prices.dropna(how="all")
        prices = _clean_prices(prices)

        # Salva cache Parquet
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path)
        logger.info("Preços salvos em cache: %s", cache_path)
        return prices

    except Exception as exc:
        logger.warning("Erro ao baixar dados online: %s. Usando dados sintéticos.", exc)
        return _generate_synthetic_prices(assets_with_bench, cfg.start_date, cfg.end_date)


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Limpeza: forward-fill, winsorização de retornos extremos."""
    # Forward-fill gaps (ex: circuit breakers B3)
    prices = prices.ffill().bfill()

    # Winsorização de retornos diários (percentis 1%-99%)
    returns = prices.pct_change()
    p01 = returns.quantile(0.01)
    p99 = returns.quantile(0.99)
    for col in returns.columns:
        returns[col] = returns[col].clip(lower=p01[col], upper=p99[col])

    # Reconstrói preços winzorizados
    prices_clean = (1 + returns.fillna(0)).cumprod() * prices.iloc[0]
    prices_clean.iloc[0] = prices.iloc[0]
    return prices_clean.dropna(how="all")


def _generate_synthetic_prices(
    tickers: List[str], start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Gera preços sintéticos realistas baseados em estatísticas históricas do mercado B3.
    Fallback quando não há conexão com internet.
    """
    logger.info("Gerando dados sintéticos realistas para %d ativos", len(tickers))
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start=start_date, end=end_date, freq="B")

    # Parâmetros realistas por tipo de ativo
    params = {
        "default":  {"mu": 0.12, "sigma": 0.25, "start": 30.0},
        "^BVSP":    {"mu": 0.08, "sigma": 0.22, "start": 115000.0},
        "BOVA11.SA":{"mu": 0.08, "sigma": 0.22, "start": 115.0},
        "PETR4.SA": {"mu": 0.15, "sigma": 0.35, "start": 32.0},
        "VALE3.SA": {"mu": 0.10, "sigma": 0.30, "start": 75.0},
        "ITUB4.SA": {"mu": 0.12, "sigma": 0.22, "start": 28.0},
        "WEGE3.SA": {"mu": 0.20, "sigma": 0.25, "start": 35.0},
        "MXRF11.SA":{"mu": 0.09, "sigma": 0.10, "start": 10.5},
        "IVVB11.SA":{"mu": 0.14, "sigma": 0.18, "start": 280.0},
        "SMAL11.SA":{"mu": 0.11, "sigma": 0.28, "start": 95.0},
    }

    n = len(dates)
    prices_dict: Dict[str, np.ndarray] = {}

    for ticker in tickers:
        p = params.get(ticker, params["default"])
        mu_daily = p["mu"] / 252
        sigma_daily = p["sigma"] / np.sqrt(252)
        returns = rng.normal(mu_daily, sigma_daily, n)
        # Adiciona correlação de mercado (beta ~0.8)
        market_shock = rng.normal(0, 0.01, n)
        returns = returns * 0.6 + market_shock * 0.4
        price = p["start"] * np.exp(np.cumsum(returns))
        prices_dict[ticker] = price

    df = pd.DataFrame(prices_dict, index=dates)
    logger.info("Dados sintéticos gerados: %d dias, %d ativos", n, len(tickers))
    return df


# ---------------------------------------------------------------------------
# Dados macro BCB/SGS (neuralops777)
# ---------------------------------------------------------------------------

def load_macro_data(config) -> Dict[str, pd.Series]:
    """
    Carrega séries macro do Banco Central do Brasil via API SGS.
    Retorna dict com séries: selic, ipca, ptax, spread.
    """
    from configs.config import QuantAIConfig
    cfg: QuantAIConfig = config

    macro_cfg = cfg.raw.get("macro", {})
    series_map = {
        "selic":  macro_cfg.get("selic_series", 432),
        "ipca":   macro_cfg.get("ipca_series", 433),
        "ptax":   macro_cfg.get("ptax_series", 1),
        "spread": macro_cfg.get("spread_series", 20786),
    }

    cache_path = cfg.data_dir / "macro.parquet"
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 24:  # TTL 24h para dados macro
            try:
                df = pd.read_parquet(cache_path)
                return {col: df[col].dropna() for col in df.columns}
            except Exception:
                pass

    result: Dict[str, pd.Series] = {}
    for name, series_id in series_map.items():
        try:
            series = _fetch_bcb_series(series_id, cfg.start_date, cfg.end_date)
            result[name] = series
            logger.info("BCB série '%s' (%d): %d observações", name, series_id, len(series))
        except Exception as exc:
            logger.warning("Erro ao buscar série BCB '%s': %s. Usando sintético.", name, exc)
            result[name] = _synthetic_macro_series(name, cfg.start_date, cfg.end_date)

    # Cache
    df_macro = pd.DataFrame(result)
    df_macro.to_parquet(cache_path)
    return result


def _fetch_bcb_series(series_id: int, start_date: str, end_date: str) -> pd.Series:
    """Busca série temporal do SGS/BCB."""
    import requests
    start_fmt = pd.to_datetime(start_date).strftime("%d/%m/%Y")
    end_fmt   = pd.to_datetime(end_date).strftime("%d/%m/%Y")
    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados"
        f"?formato=json&dataInicial={start_fmt}&dataFinal={end_fmt}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df.set_index("data")["valor"]


def _synthetic_macro_series(name: str, start_date: str, end_date: str) -> pd.Series:
    """Gera série macro sintética realista como fallback."""
    dates = pd.bdate_range(start=start_date, end=end_date, freq="ME")
    rng = np.random.default_rng(hash(name) % 2**32)
    if name == "selic":
        values = np.clip(rng.normal(10.5, 2.0, len(dates)), 2.0, 14.75)
    elif name == "ipca":
        values = np.clip(rng.normal(5.0, 1.5, len(dates)), 1.0, 12.0)
    elif name == "ptax":
        values = np.clip(rng.normal(5.2, 0.5, len(dates)), 4.0, 6.5)
    else:  # spread
        values = np.clip(rng.normal(25.0, 5.0, len(dates)), 15.0, 45.0)
    return pd.Series(values, index=dates)


# ---------------------------------------------------------------------------
# Filtro de liquidez
# ---------------------------------------------------------------------------

def filter_liquid_assets(
    prices: pd.DataFrame,
    config,
    volume_data: Optional[pd.DataFrame] = None,
) -> List[str]:
    """
    Filtra ativos com volume médio diário >= min_daily_volume_brl.
    Se volume_data não disponível, usa heurística de volatilidade.
    """
    from configs.config import QuantAIConfig
    cfg: QuantAIConfig = config
    min_vol = cfg.raw["selection"].get("min_daily_volume_brl", 1_000_000)
    benchmark = cfg.benchmark
    assets = [c for c in prices.columns if c != benchmark]

    if volume_data is not None:
        avg_vol = volume_data[assets].mean()
        liquid = avg_vol[avg_vol >= min_vol].index.tolist()
        logger.info("Filtro de liquidez: %d/%d ativos passaram", len(liquid), len(assets))
        return liquid

    # Heurística: exclui ativos com < 200 pregões ou muitos NaN
    liquid = []
    for asset in assets:
        if asset in prices.columns:
            series = prices[asset].dropna()
            if len(series) >= 200:
                liquid.append(asset)
    logger.info("Filtro de liquidez (heurístico): %d/%d ativos passaram", len(liquid), len(assets))
    return liquid