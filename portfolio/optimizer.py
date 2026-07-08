"""QuantAI Unified — optimizer.py
Otimização de portfólio usando PyPortfolioOpt com estimador Ledoit-Wolf.
Incorpora expected_returns do XGBoost quando disponíveis.
Fallback: ponderação por risco inverso (1/vol).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def optimize_portfolio(
    prices: pd.DataFrame,
    selected_assets: List[str],
    config: Dict,
    expected_returns: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Calcula pesos ótimos do portfólio.
    
    Estratégia 1 (preferida): PyPortfolioOpt Ledoit-Wolf + expected_returns ML
    Estratégia 2 (fallback): Risco inverso (1/volatilidade)
    """
    sel = config.get("selection", {})
    max_w = float(sel.get("max_weight", 0.35))
    min_w = float(sel.get("min_weight", 0.02))

    if not selected_assets:
        return pd.Series(dtype=float)

    # Filtra preços para ativos selecionados
    avail = [a for a in selected_assets if a in prices.columns]
    if not avail:
        return pd.Series(dtype=float)

    prices_sel = prices[avail].dropna(how="all")

    # Tenta otimização PyPortfolioOpt com Ledoit-Wolf
    try:
        weights = _optimize_ledoit_wolf(
            prices_sel, avail, max_w, min_w, expected_returns
        )
        logger.info("Portfólio otimizado via Ledoit-Wolf: %d ativos", len(weights))
        return weights
    except Exception as exc:
        logger.warning("Ledoit-Wolf falhou (%s). Usando risco inverso.", exc)
        return _inverse_vol_weights(prices_sel, avail, max_w)


def _optimize_ledoit_wolf(
    prices: pd.DataFrame,
    assets: List[str],
    max_w: float,
    min_w: float,
    expected_returns: Optional[pd.Series] = None,
) -> pd.Series:
    """Otimização com PyPortfolioOpt + estimador Ledoit-Wolf + regularização L2."""
    from pypfopt import EfficientFrontier, expected_returns as er, risk_models

    returns = prices.pct_change().dropna()

    # Estimador de covariância robusto Ledoit-Wolf
    cov_matrix = risk_models.CovarianceShrinkage(prices, frequency=252).ledoit_wolf()

    # Expected returns: ML ou histórico
    if expected_returns is not None and len(expected_returns) > 0:
        mu = expected_returns.reindex(assets).fillna(expected_returns.mean())
    else:
        mu = er.mean_historical_return(prices, frequency=252)

    # Otimização: Max Sharpe com regularização L2 (evita concentração)
    ef = EfficientFrontier(mu, cov_matrix, weight_bounds=(min_w, max_w))
    ef.add_objective(_l2_regularization, gamma=0.1)
    ef.max_sharpe(risk_free_rate=0.105)  # Selic ~10.5% ao ano
    cleaned = ef.clean_weights()
    weights = pd.Series({k: v for k, v in cleaned.items() if v > 0.001})

    # Normaliza para somar 1
    weights = weights / weights.sum()
    return weights


def _l2_regularization(w, gamma=0.1):
    """Regularização L2 para evitar concentração excessiva."""
    try:
        import cvxpy as cp
        return gamma * cp.sum_squares(w)
    except Exception:
        return 0


def _inverse_vol_weights(
    prices: pd.DataFrame,
    assets: List[str],
    max_w: float,
) -> pd.Series:
    """
    Ponderação por risco inverso (1/volatilidade).
    Fallback robusto que funciona sem dependências extras.
    """
    returns = prices[assets].pct_change().dropna()
    vols = returns.std() * np.sqrt(252)
    vols = vols.replace(0, np.nan).dropna()

    if vols.empty:
        n = len(assets)
        return pd.Series(1.0 / n, index=assets)

    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()

    # Aplica cap de peso máximo (iterativo)
    for _ in range(10):
        over = weights[weights > max_w]
        if over.empty:
            break
        excess = (weights[over.index] - max_w).sum()
        weights[over.index] = max_w
        under = weights[weights < max_w]
        weights[under.index] += excess * (weights[under.index] / weights[under.index].sum())

    weights = weights / weights.sum()
    return weights.round(6)


def compute_sector_exposure(
    weights: pd.Series,
    sector_map: Dict[str, str],
) -> pd.Series:
    """Calcula exposição setorial da carteira."""
    sector_weights: Dict[str, float] = {}
    for asset, weight in weights.items():
        sector = sector_map.get(asset, "Outros")
        sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
    return pd.Series(sector_weights).sort_values(ascending=False)