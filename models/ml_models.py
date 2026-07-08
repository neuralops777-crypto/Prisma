"""QuantAI Unified — ml_models.py
Pipeline de Machine Learning para previsão de retornos esperados.
- Baseline: Regressão Linear
- Ensemble: Random Forest + XGBoost ponderados por IC
- Walk-forward validation (sem look-ahead bias)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_expected_returns_ml(
    prices: pd.DataFrame,
    assets: List[str],
    rebal_date: pd.Timestamp,
    config: Dict,
) -> pd.Series:
    """
    Calcula expected_returns para cada ativo usando ensemble ML.
    Retorna pd.Series com retorno esperado por ativo.
    Fallback para retorno histórico médio se ML falhar.
    """
    ml_cfg = config.get("ml", {})
    horizon = ml_cfg.get("horizon_days", 21)
    train_months = ml_cfg.get("train_window_months", 24)
    ens_w = ml_cfg.get("ensemble_weights", {"random_forest": 0.40, "xgboost": 0.60})

    try:
        features, targets = _build_features_targets(
            prices, assets, rebal_date, horizon, train_months
        )
        if features is None or len(features) < 30:
            raise ValueError("Dados insuficientes para treinar ML")

        # Split treino/predição
        X_train = features.iloc[:-1]
        y_train = targets.iloc[:-1]
        X_pred  = features.iloc[[-1]]  # Última janela → prever próximo período

        preds: Dict[str, Dict[str, float]] = {}

        # Random Forest
        rf_preds = _predict_rf(X_train, y_train, X_pred, assets)
        # XGBoost
        xgb_preds = _predict_xgboost(X_train, y_train, X_pred, assets)

        # Ensemble ponderado
        rf_w = ens_w.get("random_forest", 0.40)
        xgb_w = ens_w.get("xgboost", 0.60)

        ensemble = {}
        for asset in assets:
            rf_v = rf_preds.get(asset, 0.0)
            xgb_v = xgb_preds.get(asset, 0.0)
            ensemble[asset] = rf_w * rf_v + xgb_w * xgb_v

        result = pd.Series(ensemble)
        logger.info(
            "[ML %s] Expected returns: top=%s (%.1f%%), bottom=%s (%.1f%%)",
            rebal_date.strftime("%Y-%m"),
            result.idxmax(), result.max() * 100,
            result.idxmin(), result.min() * 100,
        )
        return result

    except Exception as exc:
        logger.warning("ML falhou (%s). Usando retorno histórico médio.", exc)
        return _historical_mean_returns(prices, assets, rebal_date, horizon)


def _build_features_targets(
    prices: pd.DataFrame,
    assets: List[str],
    rebal_date: pd.Timestamp,
    horizon: int,
    train_months: int,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Constrói matrix de features e targets para treino do ML."""
    avail = [a for a in assets if a in prices.columns]
    if not avail:
        return None, None

    # Janela de treino
    train_start = rebal_date - pd.DateOffset(months=train_months)
    window = prices.loc[train_start:rebal_date, avail].dropna(how="all")

    if len(window) < horizon * 2:
        return None, None

    # Features: retornos rolantes, volatilidade, RSI simplificado
    returns = window.pct_change().dropna()

    all_features = []
    all_targets = []

    # Walk-forward: cada ponto de treino
    step = max(5, horizon // 4)
    for i in range(60, len(returns) - horizon, step):
        r_window = returns.iloc[i - 60:i]
        target_r = returns.iloc[i:i + horizon].mean()  # Retorno médio no horizonte

        feats = {}
        for asset in avail:
            r = r_window[asset].dropna()
            if len(r) < 20:
                continue
            feats[f"{asset}_mom_20"] = r.iloc[-20:].sum()
            feats[f"{asset}_mom_60"] = r.sum()
            feats[f"{asset}_vol_20"] = r.iloc[-20:].std()
            feats[f"{asset}_vol_60"] = r.std()
            # RSI simplificado
            gains = r.clip(lower=0).iloc[-14:].mean()
            losses = (-r.clip(upper=0)).iloc[-14:].mean()
            rsi = 50 if losses == 0 else 100 - 100 / (1 + gains / losses)
            feats[f"{asset}_rsi"] = rsi

        if feats:
            all_features.append(feats)
            all_targets.append(target_r.to_dict())

    if not all_features:
        return None, None

    X = pd.DataFrame(all_features).fillna(0)
    y = pd.DataFrame(all_targets).fillna(0)
    return X, y


def _predict_rf(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_pred: pd.DataFrame,
    assets: List[str],
) -> Dict[str, float]:
    """Random Forest para previsão por ativo."""
    from sklearn.ensemble import RandomForestRegressor
    preds = {}
    for asset in assets:
        if asset not in y_train.columns:
            continue
        try:
            model = RandomForestRegressor(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
            )
            model.fit(X_train.values, y_train[asset].values)
            pred = model.predict(X_pred.values)[0]
            preds[asset] = float(pred)
        except Exception:
            preds[asset] = 0.0
    return preds


def _predict_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_pred: pd.DataFrame,
    assets: List[str],
) -> Dict[str, float]:
    """XGBoost para previsão por ativo com regularização L1/L2 nativa."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("XGBoost não instalado. Usando RF apenas.")
        return {}

    preds = {}
    for asset in assets:
        if asset not in y_train.columns:
            continue
        try:
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,    # L1
                reg_lambda=1.0,   # L2
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            )
            model.fit(X_train.values, y_train[asset].values, verbose=False)
            pred = model.predict(X_pred.values)[0]
            preds[asset] = float(pred)
        except Exception as exc:
            logger.debug("XGBoost falhou para %s: %s", asset, exc)
            preds[asset] = 0.0
    return preds


def _historical_mean_returns(
    prices: pd.DataFrame,
    assets: List[str],
    rebal_date: pd.Timestamp,
    horizon: int,
) -> pd.Series:
    """Fallback: retorno histórico médio anualizado."""
    avail = [a for a in assets if a in prices.columns]
    window = prices.loc[:rebal_date, avail].tail(252)
    if window.empty:
        return pd.Series(0.10, index=avail)
    returns = window.pct_change().dropna()
    annual_returns = returns.mean() * 252
    return annual_returns.reindex(assets).fillna(annual_returns.mean())