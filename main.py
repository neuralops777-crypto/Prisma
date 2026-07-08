"""QuantAI Unified — main.py
Pipeline CLI completo de 10 etapas.
Execução: python main.py [--config configs/default.yml] [--use-agent] [--no-ml]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quantai")


def main():
    parser = argparse.ArgumentParser(description="QuantAI Unified 2026 — Pipeline Completo")
    parser.add_argument("--config",    default="configs/default.yml", help="Config YAML")
    parser.add_argument("--use-agent", action="store_true",           help="Usa agente OpenAI real")
    parser.add_argument("--no-ml",     action="store_true",           help="Desativa ML/XGBoost")
    parser.add_argument("--update",    action="store_true",           help="Força download de dados")
    args = parser.parse_args()

    t0 = time.time()
    print("\n" + "="*60)
    print("  ⚡ QUANTAI UNIFIED 2026 — Pipeline de 10 Etapas")
    print("="*60)

    # ── Etapa 1: Configuração ──────────────────────────────────
    print("\n[1/10] 📋 Carregando configuração...")
    from configs.config import load_config
    config = load_config(ROOT / args.config)
    if args.no_ml:
        config.raw["ml"] = config.raw.get("ml", {})
        config.raw["ml"]["ensemble_weights"] = {"random_forest": 0.0, "xgboost": 0.0}
    print(f"      Ativos: {len(config.assets)} | Período: {config.start_date} → {config.end_date}")

    # ── Etapa 2: Coleta de dados de mercado ───────────────────
    print("\n[2/10] 📥 Coletando dados de mercado (yfinance/cache)...")
    from data.data import load_prices, load_macro_data, filter_liquid_assets
    prices = load_prices(config, force_download=args.update)
    print(f"      {len(prices.columns)} séries | {len(prices)} pregões")

    # ── Etapa 3: Dados macro BCB/SGS ──────────────────────────
    print("\n[3/10] 🏦 Coletando dados macro BCB/SGS...")
    try:
        macro_data = load_macro_data(config)
        print(f"      Séries macro: {list(macro_data.keys())}")
    except Exception as e:
        macro_data = {}
        print(f"      ⚠️  Macro indisponível: {e}")

    # ── Etapa 4: Filtro de liquidez ───────────────────────────
    print("\n[4/10] 💧 Aplicando filtro de liquidez...")
    liquid_assets = filter_liquid_assets(prices, config)
    print(f"      {len(liquid_assets)}/{len(config.assets)} ativos com liquidez adequada")

    # ── Etapa 5: Features técnicas (internalizado no backtest) ─
    print("\n[5/10] 🔧 Features técnicas calculadas no motor de backtest")
    print("      Momentum 3M/6M/12M, RSI(14), ATR, Bollinger, R²-Tendência")

    # ── Etapa 6: Features ML (opcional) ──────────────────────
    print("\n[6/10] 🧠 Modelos ML:", end=" ")
    if args.no_ml:
        print("desativado (--no-ml)")
    else:
        print("XGBoost + Random Forest (walk-forward)")

    # ── Etapa 7: Otimização de portfólio ──────────────────────
    print("\n[7/10] ⚙️  Otimizador: Ledoit-Wolf + L2 (PyPortfolioOpt)")

    # ── Etapa 8: Backtest completo ─────────────────────────────
    print("\n[8/10] 📊 Executando backtest...")
    from backtest_engine.backtest import run_backtest
    result = run_backtest(prices, config.raw)

    m  = result.metrics
    bm = result.benchmark_metrics
    alpha = (m.get("cagr", 0) - bm.get("cagr", 0)) * 100
    print(f"\n      ┌─── Resultados ───────────────────────────────┐")
    print(f"      │  CAGR:     {m.get('cagr',0)*100:6.1f}% (benchmark: {bm.get('cagr',0)*100:.1f}%)")
    print(f"      │  Alpha:    {alpha:+6.1f}pp/ano")
    print(f"      │  Sharpe:   {m.get('sharpe',0):6.2f} (benchmark: {bm.get('sharpe',0):.2f})")
    print(f"      │  Drawdown: {m.get('max_drawdown',0)*100:6.1f}% (benchmark: {bm.get('max_drawdown',0)*100:.1f}%)")
    print(f"      │  Sortino:  {m.get('sortino',0):6.2f}")
    print(f"      │  Calmar:   {m.get('calmar',0):6.2f}")
    print(f"      │  VaR 95%:  {m.get('var_95_daily',0)*100:6.1f}%/dia")
    print(f"      │  Capital:  R$ {m.get('final_equity',0):,.0f}")
    print(f"      │  Rebalanceamentos: {len(result.rebalance_dates)}")
    print(f"      └─────────────────────────────────────────────┘")

    # ── Etapa 9: Agente IA ────────────────────────────────────
    print(f"\n[9/10] 🤖 Agente IA: {'gerando parecer...' if args.use_agent else 'fallback auditado'}")
    agent_report = None
    if args.use_agent:
        from reports.reporting import save_outputs
        files = save_outputs(result, config.raw, config.reports_dir, use_genai=True)
    else:
        from reports.reporting import save_outputs
        files = save_outputs(result, config.raw, config.reports_dir, use_genai=False)

    # ── Etapa 10: Relatórios ──────────────────────────────────
    print(f"\n[10/10] 📝 Relatórios gerados em: {config.reports_dir}")
    for name, path in files.items():
        if path and Path(path).exists():
            size_kb = Path(path).stat().st_size / 1024
            print(f"        ✅ {name}: {Path(path).name} ({size_kb:.1f} KB)")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  ✅ Pipeline concluído em {elapsed:.1f}s")
    print(f"  📊 Dashboard: streamlit run app.py")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())