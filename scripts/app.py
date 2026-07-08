"""QuantAI Unified — app.py
Dashboard Streamlit interativo com:
- Sliders de parâmetros em tempo real
- Curva de capital, drawdown, pesos setoriais
- Auditoria IA em JSON
- Métricas completas com comparação ao benchmark
- Análise de regime de mercado
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# Configura secrets do Streamlit Cloud
for key in ["OPENAI_API_KEY", "OPENAI_MODEL"]:
    if key in st.secrets:
        os.environ.setdefault(key, st.secrets[key])

from configs.config import load_config
from data.data import load_prices, load_macro_data, filter_liquid_assets
from backtest_engine.backtest import run_backtest
from portfolio.optimizer import compute_sector_exposure
from reports.reporting import build_markdown_report, save_outputs

# ─────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="QuantAI Unified 2026",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ QuantAI Unified 2026")
st.caption(
    "Robo quantitativo B3 — XGBoost + Ledoit-Wolf + Agente OpenAI auditável | "
    "Gabriel18182 × neuralops777 — Versão Unificada"
)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
config_path = ROOT / "configs" / "default.yml"
config = load_config(config_path)

with st.sidebar:
    st.header("⚙️ Execução")
    force_download = st.toggle("🔄 Atualizar dados", value=False)
    use_agent = st.toggle("🤖 Usar agente OpenAI", value=False)
    use_ml = st.toggle("🧠 Usar ML (XGBoost)", value=True)
    st.caption("Sem OPENAI_API_KEY → fallback auditado determinístico.")

    st.divider()
    st.header("📊 Parâmetros")

    all_assets = config.assets
    top_n = st.slider("Nº de ativos", 3, min(len(all_assets), 15), config.top_n)
    max_weight = st.slider("Peso máximo", 0.10, 0.60, config.max_weight, step=0.05,
                           format="%.0f%%", help="Limite superior de alocação por ativo")
    target_vol = st.slider("Vol alvo anual", 0.05, 0.40,
                           config.raw["selection"].get("volatility_target_annual", 0.18),
                           step=0.01, format="%.0f%%")
    tc_bps = st.slider("Custo transação (bps)", 0, 30,
                       int(config.transaction_cost_bps), step=1)
    slip_bps = st.slider("Slippage (bps)", 0, 20, int(config.slippage_bps), step=1)

    st.divider()
    st.header("📅 Período")
    start_date = st.date_input("Data inicial", pd.to_datetime(config.start_date))
    end_date   = st.date_input("Data final",   pd.to_datetime(config.end_date))

    run_btn = st.button("▶️ Executar Backtest", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# Config dinâmica
# ─────────────────────────────────────────────
run_config = config.raw.copy()
run_config["selection"] = dict(config.raw["selection"])
run_config["selection"]["top_n"] = top_n
run_config["selection"]["max_weight"] = max_weight
run_config["selection"]["volatility_target_annual"] = target_vol
run_config["backtest"] = dict(config.raw["backtest"])
run_config["backtest"]["transaction_cost_bps"] = tc_bps
run_config["backtest"]["slippage_bps"] = slip_bps
run_config["backtest"]["start_date"] = str(start_date)
run_config["backtest"]["end_date"]   = str(end_date)

# Desativa ML se não solicitado
if not use_ml:
    run_config["ml"] = run_config.get("ml", {})
    run_config["ml"]["ensemble_weights"] = {"random_forest": 0.0, "xgboost": 0.0}

# ─────────────────────────────────────────────
# Cache de resultado na sessão
# ─────────────────────────────────────────────
if "result" not in st.session_state or run_btn:
    with st.spinner("🔄 Carregando dados e executando backtest..."):
        prices = load_prices(config, force_download=force_download)
        result = run_backtest(prices, run_config)
        st.session_state["result"] = result
        st.session_state["prices"] = prices
        st.session_state["run_config"] = run_config

result = st.session_state["result"]
prices = st.session_state["prices"]

# ─────────────────────────────────────────────
# KPIs principais
# ─────────────────────────────────────────────
m  = result.metrics
bm = result.benchmark_metrics

def fpct(v): return f"{v*100:.1f}%" if v is not None else "N/A"
def fn(v, d=2): return f"{v:.{d}f}" if v is not None else "N/A"

st.subheader("📈 KPIs de Performance")
cols = st.columns(6)
alpha = (m.get("cagr", 0) - bm.get("cagr", 0)) * 100
cols[0].metric("CAGR",         fpct(m.get("cagr")),  f"vs {fpct(bm.get('cagr'))} benchmark")
cols[1].metric("Sharpe",       fn(m.get("sharpe")),   f"vs {fn(bm.get('sharpe'))} benchmark")
cols[2].metric("Sortino",      fn(m.get("sortino")),  "Meta: > 1.5")
cols[3].metric("Max Drawdown", fpct(m.get("max_drawdown")), f"vs {fpct(bm.get('max_drawdown'))}")
cols[4].metric("Alpha p.a.",   f"+{alpha:.1f}pp" if alpha >= 0 else f"{alpha:.1f}pp")
cols[5].metric("Capital Final", f"R$ {m.get('final_equity',0):,.0f}")

st.divider()

# ─────────────────────────────────────────────
# Gráficos principais
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Curva de Capital", "📉 Drawdown", "🗂️ Carteira", "📊 Métricas Detalhadas", "🤖 IA & Auditoria"]
)

with tab1:
    st.subheader("Curva de Capital vs Benchmark")
    equity = result.equity_curve.copy()
    equity.index = pd.to_datetime(equity.index)
    st.line_chart(equity, color=["#1f77b4", "#ff7f0e"])
    
    # Retorno mensal
    monthly = equity.resample("ME").last().pct_change().dropna() * 100
    st.subheader("Retorno Mensal (%)")
    st.bar_chart(monthly["strategy"])

with tab2:
    st.subheader("Drawdown")
    roll_max = equity["strategy"].cummax()
    drawdown = (equity["strategy"] - roll_max) / roll_max * 100
    bench_max = equity["benchmark"].cummax()
    bench_dd  = (equity["benchmark"] - bench_max) / bench_max * 100
    dd_df = pd.DataFrame({"Estratégia": drawdown, "Benchmark": bench_dd})
    st.area_chart(dd_df)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Max Drawdown",    fpct(m.get("max_drawdown")))
    col2.metric("DD Duration (d)", m.get("max_drawdown_duration_days", "N/A"))
    col3.metric("Calmar Ratio",    fn(m.get("calmar")))

with tab3:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Pesos da Carteira ao Longo do Tempo")
        if not result.weights.empty:
            top_assets_plot = result.weights.mean().nlargest(8).index
            st.area_chart(result.weights[top_assets_plot].fillna(0))
    with right:
        st.subheader("Última Carteira")
        if not result.weights.empty:
            last_w = result.weights.iloc[-1].sort_values(ascending=False)
            last_w = last_w[last_w > 0.001]
            display_df = pd.DataFrame({
                "Ativo": last_w.index,
                "Peso (%)": (last_w.values * 100).round(1),
                "Setor": [run_config.get("sector_map", {}).get(a, "—") for a in last_w.index],
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.subheader("Exposição Setorial")
        if not result.weights.empty and run_config.get("sector_map"):
            last_w_series = result.weights.iloc[-1][result.weights.iloc[-1] > 0.001]
            sector_exp = compute_sector_exposure(last_w_series, run_config.get("sector_map", {}))
            st.bar_chart(sector_exp)

with tab4:
    st.subheader("Métricas Completas")
    metrics_table = {
        "CAGR":              [fpct(m.get("cagr")),        fpct(bm.get("cagr")),        "> 15% a.a."],
        "Retorno Total":     [fpct(m.get("total_return")), fpct(bm.get("total_return")), "> 50% (3a)"],
        "Sharpe Ratio":      [fn(m.get("sharpe")),         fn(bm.get("sharpe")),         "> 1.0"],
        "Sortino Ratio":     [fn(m.get("sortino")),        "—",                           "> 1.5"],
        "Max Drawdown":      [fpct(m.get("max_drawdown")), fpct(bm.get("max_drawdown")),  "< -20%"],
        "DD Duration (dias)":[str(m.get("max_drawdown_duration_days","—")), "—",          "< 180d"],
        "Calmar Ratio":      [fn(m.get("calmar")),         "—",                           "> 0.5"],
        "Volatilidade":      [fpct(m.get("volatility")),   fpct(bm.get("volatility")),    "< 20%"],
        "VaR 95% (1d)":      [fpct(m.get("var_95_daily")), "—",                           "> -3%"],
        "Win Rate":          [fpct(m.get("win_rate")),     fpct(bm.get("win_rate")),      "> 52%"],
        "Hit Rate Mensal":   [fpct(m.get("monthly_hit_rate")), "—",                       "> 55%"],
    }
    df_metrics = pd.DataFrame(metrics_table, index=["Estratégia", "Benchmark", "Meta"]).T
    st.dataframe(df_metrics, use_container_width=True)

    if not result.trades.empty:
        st.subheader("Histórico de Rebalanceamentos")
        st.dataframe(result.trades.tail(12), use_container_width=True)

with tab5:
    st.subheader("🤖 Agente IA & Auditoria")

    # Gera/carrega relatório do agente
    audit_path = config.reports_dir / "agent_audit.json"
    agent_audit = {}
    if audit_path.exists():
        try:
            agent_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    col1, col2 = st.columns(2)
    col1.metric("Provider",    agent_audit.get("provider", "openai"))
    col1.metric("Modelo",      agent_audit.get("model", config.ai_model))
    col2.metric("Status",      agent_audit.get("status", "não gerado"))
    col2.metric("Custo Estim.",f"US$ {agent_audit.get('estimated_cost_usd', 0):.4f}")

    st.subheader("Auditoria JSON")
    st.json(agent_audit)

    if use_agent and run_btn:
        with st.spinner("🤖 Gerando parecer do agente OpenAI..."):
            files = save_outputs(result, run_config, config.reports_dir, use_genai=True)
            st.success("Relatório do agente gerado!")

    st.subheader("Relatório Executivo")
    st.markdown(build_markdown_report(result, run_config, use_genai=False))

    # Download de arquivos
    st.divider()
    st.subheader("📥 Downloads")
    col1, col2, col3 = st.columns(3)

    eq_path = config.reports_dir / "equity_curve.csv"
    if eq_path.exists():
        col1.download_button(
            "📊 Equity Curve (CSV)",
            eq_path.read_bytes(),
            "equity_curve.csv",
            "text/csv",
        )

    metrics_path = config.reports_dir / "metrics.json"
    if metrics_path.exists():
        col2.download_button(
            "📋 Métricas (JSON)",
            metrics_path.read_bytes(),
            "metrics.json",
            "application/json",
        )

    exec_path = config.reports_dir / "executive_report.md"
    if exec_path.exists():
        col3.download_button(
            "📄 Relatório (MD)",
            exec_path.read_bytes(),
            "executive_report.md",
            "text/markdown",
        )

# ─────────────────────────────────────────────
# Regimes de mercado
# ─────────────────────────────────────────────
if result.regime_history:
    with st.expander("🌊 Histórico de Regime de Mercado"):
        regime_df = pd.Series(result.regime_history)
        regime_counts = regime_df.value_counts()
        cols = st.columns(len(regime_counts))
        colors_map = {"bull": "🟢", "lateral": "🟡", "bear": "🔴"}
        for i, (regime, count) in enumerate(regime_counts.items()):
            cols[i].metric(
                f"{colors_map.get(regime, '⚪')} {regime.capitalize()}",
                f"{count} rebalanceamentos",
                f"{count/len(regime_df)*100:.0f}% do tempo"
            )

st.caption(
    "QuantAI Unified v2.0.0 | Fusão Gabriel18182/QuantAI + neuralops777/quant-ai-itau | "
    "⚠️ Apenas educacional — não constitui recomendação de investimento."
)