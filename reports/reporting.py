"""QuantAI Unified — reporting.py
Geração de relatórios: CSV, JSON, Markdown, HTML, PDF (ReportLab).
Inclui gráficos, métricas formatadas e auditoria de IA.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def save_outputs(
    result,
    config: Dict,
    reports_dir: Path,
    use_genai: bool = False,
    sector_exposure: Optional[Dict] = None,
) -> Dict[str, Path]:
    """
    Salva todos os artefatos do backtest no diretório de relatórios.
    Retorna dict com caminhos de cada arquivo gerado.
    """
    from backtest_engine.agent import generate_agent_report

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    # 1. Equity curve CSV
    equity_path = reports_dir / "equity_curve.csv"
    result.equity_curve.to_csv(equity_path)
    files["equity_curve"] = equity_path

    # 2. Métricas JSON
    metrics_path = reports_dir / "metrics.json"
    metrics_data = {
        "strategy": result.metrics,
        "benchmark": result.benchmark_metrics,
        "generated_at": datetime.now().isoformat(),
        "config_summary": {
            "period": f"{config['backtest']['start_date']} → {config['backtest']['end_date']}",
            "top_n": config["selection"]["top_n"],
            "cost_bps": config["backtest"]["transaction_cost_bps"],
            "slippage_bps": config["backtest"].get("slippage_bps", 5),
        }
    }
    metrics_path.write_text(json.dumps(metrics_data, indent=2, default=str), encoding="utf-8")
    files["metrics"] = metrics_path

    # 3. Positions CSV
    if not result.weights.empty:
        pos_path = reports_dir / "positions.csv"
        result.weights.to_csv(pos_path)
        files["positions"] = pos_path

    # 4. Trades CSV
    if not result.trades.empty:
        trades_path = reports_dir / "trades.csv"
        result.trades.to_csv(trades_path, index=False)
        files["trades"] = trades_path

    # 5. Regime history JSON
    regime_path = reports_dir / "regime_history.json"
    regime_path.write_text(json.dumps(result.regime_history, indent=2), encoding="utf-8")
    files["regime"] = regime_path

    # 6. Agente IA
    agent_report = None
    if use_genai:
        try:
            weights_dict = {}
            if not result.weights.empty:
                last_w = result.weights.iloc[-1]
                weights_dict = last_w[last_w > 0.001].to_dict()

            agent_report = generate_agent_report(
                result.metrics,
                result.benchmark_metrics,
                weights_dict,
                config,
                sector_exposure,
            )

            audit_path = reports_dir / "agent_audit.json"
            audit_path.write_text(json.dumps(agent_report.audit, indent=2), encoding="utf-8")
            files["agent_audit"] = audit_path

            ar_path = reports_dir / "agent_report.md"
            ar_path.write_text(_format_agent_report_md(agent_report), encoding="utf-8")
            files["agent_report"] = ar_path

        except Exception as exc:
            logger.warning("Falha ao gerar relatório do agente: %s", exc)
    else:
        # Gera audit JSON de fallback mesmo sem use_genai
        audit_fallback = {
            "provider": "openai",
            "model": config.get("ai", {}).get("model", "gpt-4o-mini"),
            "used_live_agent": False,
            "agent_sdk_used": False,
            "estimated_cost_usd": 0.0,
            "tokens_used": 0,
            "status": "skipped_by_user",
            "timestamp": datetime.now().isoformat(),
        }
        audit_path = reports_dir / "agent_audit.json"
        audit_path.write_text(json.dumps(audit_fallback, indent=2), encoding="utf-8")
        files["agent_audit"] = audit_path

    # 7. Relatório executivo Markdown
    exec_path = reports_dir / "executive_report.md"
    exec_path.write_text(
        build_markdown_report(result, config, use_genai=use_genai, agent_report=agent_report),
        encoding="utf-8",
    )
    files["executive_report"] = exec_path

    # 8. Gráficos
    try:
        fig_paths = _generate_charts(result, reports_dir)
        files.update(fig_paths)
    except Exception as exc:
        logger.warning("Erro ao gerar gráficos: %s", exc)

    # 9. PDF profissional (ReportLab)
    try:
        pdf_path = _generate_pdf_report(result, config, reports_dir, agent_report)
        files["pdf_report"] = pdf_path
    except Exception as exc:
        logger.warning("Erro ao gerar PDF: %s", exc)

    # 10. HTML dashboard estático
    try:
        html_path = _generate_html_report(result, config, reports_dir, agent_report)
        files["html_report"] = html_path
    except Exception as exc:
        logger.warning("Erro ao gerar HTML: %s", exc)

    logger.info("Relatórios gerados em: %s (%d arquivos)", reports_dir, len(files))
    return files


def build_markdown_report(
    result,
    config: Dict,
    use_genai: bool = False,
    agent_report=None,
) -> str:
    """Constrói relatório executivo em Markdown."""
    m = result.metrics
    bm = result.benchmark_metrics
    bt = config.get("backtest", {})

    def fmt_pct(v, signed=False):
        if v is None: return "N/A"
        prefix = "+" if signed and v > 0 else ""
        return f"{prefix}{v*100:.1f}%"

    def fmt_n(v, d=2):
        return f"{v:.{d}f}" if v is not None else "N/A"

    alpha = (m.get("cagr", 0) - bm.get("cagr", 0))

    lines = [
        "# Relatório Executivo — QuantAI Unified 2026",
        f"**Gerado em:** {datetime.now().strftime('%d/%m/%Y %H:%M')}  ",
        f"**Período:** {bt.get('start_date')} → {bt.get('end_date')}  ",
        f"**Capital Inicial:** R$ {bt.get('initial_capital', 100000):,.0f}  ",
        "",
        "---",
        "",
        "## Métricas de Performance",
        "",
        "| Métrica | Estratégia | Benchmark | Alpha |",
        "|---------|-----------|-----------|-------|",
        f"| **CAGR** | {fmt_pct(m.get('cagr'))} | {fmt_pct(bm.get('cagr'))} | {fmt_pct(alpha, signed=True)} |",
        f"| **Sharpe Ratio** | {fmt_n(m.get('sharpe'))} | {fmt_n(bm.get('sharpe'))} | — |",
        f"| **Sortino Ratio** | {fmt_n(m.get('sortino'))} | — | — |",
        f"| **Max Drawdown** | {fmt_pct(m.get('max_drawdown'))} | {fmt_pct(bm.get('max_drawdown'))} | — |",
        f"| **Calmar Ratio** | {fmt_n(m.get('calmar'))} | — | — |",
        f"| **Volatilidade** | {fmt_pct(m.get('volatility'))} | {fmt_pct(bm.get('volatility'))} | — |",
        f"| **VaR 95% (1d)** | {fmt_pct(m.get('var_95_daily'))} | — | — |",
        f"| **Win Rate** | {fmt_pct(m.get('win_rate'))} | — | — |",
        f"| **Capital Final** | R$ {m.get('final_equity', 0):,.0f} | — | — |",
        "",
        "---",
        "",
    ]

    # Última carteira
    if not result.weights.empty:
        last_w = result.weights.iloc[-1].sort_values(ascending=False)
        last_w = last_w[last_w > 0.001]
        lines += [
            "## Última Carteira",
            "",
            "| Ativo | Peso |",
            "|-------|------|",
        ]
        for asset, w in last_w.items():
            lines.append(f"| {asset} | {w*100:.1f}% |")
        lines.append("")

    # Rebalanceamentos
    n_rebal = len(result.rebalance_dates)
    lines += [
        "## Estatísticas Operacionais",
        "",
        f"- **Rebalanceamentos realizados:** {n_rebal}",
    ]
    if not result.trades.empty:
        avg_turnover = result.trades.get("turnover", pd.Series([0])).mean()
        total_cost = result.trades.get("cost_pct", pd.Series([0])).sum()
        lines += [
            f"- **Turnover médio:** {avg_turnover*100:.1f}%",
            f"- **Custo total estimado:** {total_cost:.2f}%",
        ]
    lines.append("")

    # Parecer do agente
    if agent_report:
        lines += [
            "---",
            "",
            "## Parecer do Agente IA",
            "",
            f"*Modelo: {agent_report.audit.get('model', 'N/A')} | "
            f"Status: {agent_report.audit.get('status', 'N/A')} | "
            f"Custo: US${agent_report.audit.get('estimated_cost_usd', 0):.4f}*",
            "",
            "### Resumo Executivo",
            agent_report.summary,
            "",
            "### Análise de Riscos",
            agent_report.risk_analysis,
            "",
            "### Recomendações",
            agent_report.recommendations,
            "",
            "### Perspectiva de Mercado",
            agent_report.market_outlook,
        ]

    lines += [
        "",
        "---",
        "",
        "*Aviso Legal: Este relatório é estritamente educacional. "
        "Resultados de backtest não garantem performance futura. "
        "Nenhuma informação constitui recomendação de investimento.*",
    ]

    return "\n".join(lines)


def _format_agent_report_md(agent_report) -> str:
    return f"""# Parecer do Agente IA — QuantAI Unified

**Modelo:** {agent_report.audit.get('model', 'N/A')}
**Status:** {agent_report.audit.get('status', 'N/A')}
**Tokens:** {agent_report.audit.get('tokens_used', 0):,}
**Custo estimado:** US${agent_report.audit.get('estimated_cost_usd', 0):.6f}
**Timestamp:** {agent_report.audit.get('timestamp', 'N/A')}

---

## Resumo Executivo

{agent_report.summary}

## Análise de Riscos

{agent_report.risk_analysis}

## Recomendações

{agent_report.recommendations}

## Perspectiva de Mercado

{agent_report.market_outlook}
"""


def _generate_charts(result, reports_dir: Path) -> Dict[str, Path]:
    """Gera gráficos: equity curve, drawdown, alocação setorial."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    files = {}
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    fig.suptitle("QuantAI Unified 2026 — Análise de Performance", fontsize=14, fontweight="bold")

    # 1. Equity curve
    ax = axes[0]
    equity = result.equity_curve
    ax.plot(equity.index, equity["strategy"], label="Estratégia", color="#1f77b4", linewidth=2)
    ax.plot(equity.index, equity["benchmark"], label="Benchmark (^BVSP)", color="#ff7f0e",
            linewidth=1.5, linestyle="--")
    ax.set_title("Curva de Capital")
    ax.set_ylabel("Capital (R$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    # 2. Drawdown
    ax = axes[1]
    roll_max = equity["strategy"].cummax()
    drawdown = (equity["strategy"] - roll_max) / roll_max
    ax.fill_between(equity.index, drawdown, 0, color="#d62728", alpha=0.6, label="Drawdown")
    bench_roll = equity["benchmark"].cummax()
    bench_dd = (equity["benchmark"] - bench_roll) / bench_roll
    ax.fill_between(equity.index, bench_dd, 0, color="#ff7f0e", alpha=0.3, label="DD Benchmark")
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 3. Pesos da carteira (área)
    ax = axes[2]
    if not result.weights.empty:
        top_assets = result.weights.mean().nlargest(8).index
        weights_plot = result.weights[top_assets].fillna(0)
        ax.stackplot(
            weights_plot.index,
            [weights_plot[a] for a in top_assets],
            labels=top_assets,
            alpha=0.8,
        )
        ax.set_title("Composição da Carteira ao Longo do Tempo")
        ax.set_ylabel("Peso (%)")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    chart_path = reports_dir / "figures"
    chart_path.mkdir(exist_ok=True)

    full_path = chart_path / "performance_analysis.png"
    fig.savefig(full_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    files["performance_chart"] = full_path

    # Gráfico individual equity
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(equity.index, equity["strategy"], label="Estratégia", color="#1f77b4", linewidth=2)
    ax2.plot(equity.index, equity["benchmark"], label="IBOVESPA", color="#ff7f0e", linestyle="--")
    ax2.set_title("QuantAI — Curva de Capital vs Benchmark")
    ax2.set_ylabel("Capital (R$)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    eq_path = chart_path / "equity_curve.png"
    fig2.savefig(eq_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    files["equity_chart"] = eq_path

    return files


def _generate_pdf_report(result, config, reports_dir: Path, agent_report=None) -> Path:
    """Gera PDF profissional usando ReportLab."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, PageBreak,
        )
    except ImportError:
        logger.warning("ReportLab não instalado. Pulando geração de PDF.")
        return None

    pdf_path = reports_dir / "final_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=18, textColor=colors.HexColor("#1a3a5c"))
    h2_style    = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#1f77b4"))
    body_style  = styles["BodyText"]
    small_style = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8, textColor=colors.grey)

    m  = result.metrics
    bm = result.benchmark_metrics
    bt = config.get("backtest", {})

    def fpct(v): return f"{v*100:.1f}%" if v is not None else "N/A"
    def fn(v, d=2): return f"{v:.{d}f}" if v is not None else "N/A"
    alpha = (m.get("cagr", 0) - bm.get("cagr", 0)) * 100

    story = []

    # Cabeçalho
    story.append(Paragraph("QUANT AI UNIFIED 2026", title_style))
    story.append(Paragraph("Relatório Técnico de Performance", styles["Heading3"]))
    story.append(Paragraph(
        f"Período: {bt.get('start_date')} → {bt.get('end_date')} | "
        f"Capital: R$ {bt.get('initial_capital', 100000):,.0f} | "
        f"Gerado: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        small_style
    ))
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1f77b4")))
    story.append(Spacer(1, 0.5*cm))

    # Sumário de métricas
    story.append(Paragraph("Métricas de Performance", h2_style))

    metrics_data = [
        ["Métrica", "Estratégia", "Benchmark", "Alpha/Diff"],
        ["CAGR",         fpct(m.get("cagr")),       fpct(bm.get("cagr")),       f"+{alpha:.1f}pp"],
        ["Sharpe Ratio", fn(m.get("sharpe")),        fn(bm.get("sharpe")),       "—"],
        ["Sortino Ratio",fn(m.get("sortino")),       "—",                         "—"],
        ["Max Drawdown", fpct(m.get("max_drawdown")),fpct(bm.get("max_drawdown")),"—"],
        ["Calmar Ratio", fn(m.get("calmar")),        "—",                         "—"],
        ["Volatilidade", fpct(m.get("volatility")),  fpct(bm.get("volatility")),  "—"],
        ["VaR 95% (1d)", fpct(m.get("var_95_daily")),"—",                         "—"],
        ["Win Rate",     fpct(m.get("win_rate")),    "—",                         "—"],
        ["Capital Final",f"R$ {m.get('final_equity',0):,.0f}", "—",               "—"],
    ]

    tbl = Table(metrics_data, colWidths=[4.5*cm, 3.5*cm, 3.5*cm, 3*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.5*cm))

    # Última carteira
    if not result.weights.empty:
        story.append(Paragraph("Última Carteira", h2_style))
        last_w = result.weights.iloc[-1].sort_values(ascending=False)
        last_w = last_w[last_w > 0.001]
        alloc_data = [["Ativo", "Peso", "Setor"]]
        sector_map = config.get("sector_map", {})
        for asset, w in last_w.items():
            alloc_data.append([asset, f"{w*100:.1f}%", sector_map.get(asset, "—")])
        alloc_tbl = Table(alloc_data, colWidths=[5*cm, 3*cm, 5*cm])
        alloc_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1f77b4")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
            ("PADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(alloc_tbl)
        story.append(Spacer(1, 0.5*cm))

    # Parecer da IA
    if agent_report:
        story.append(PageBreak())
        story.append(Paragraph("Parecer do Agente IA", h2_style))
        ai_info = (
            f"Modelo: {agent_report.audit.get('model','N/A')} | "
            f"Status: {agent_report.audit.get('status','N/A')} | "
            f"Custo: US${agent_report.audit.get('estimated_cost_usd',0):.4f} | "
            f"Tokens: {agent_report.audit.get('tokens_used',0):,}"
        )
        story.append(Paragraph(ai_info, small_style))
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("Resumo Executivo", styles["Heading3"]))
        story.append(Paragraph(agent_report.summary, body_style))
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("Análise de Riscos", styles["Heading3"]))
        story.append(Paragraph(agent_report.risk_analysis, body_style))
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("Recomendações", styles["Heading3"]))
        story.append(Paragraph(agent_report.recommendations, body_style))

    # Rodapé legal
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Paragraph(
        "Aviso Legal: Este documento é educacional. Resultados de backtest não garantem performance futura. "
        "Nenhuma informação constitui recomendação de investimento.",
        small_style,
    ))

    doc.build(story)
    logger.info("PDF gerado: %s", pdf_path)
    return pdf_path


def _generate_html_report(result, config, reports_dir: Path, agent_report=None) -> Path:
    """Gera dashboard HTML estático com métricas e gráfico."""
    m  = result.metrics
    bm = result.benchmark_metrics
    bt = config.get("backtest", {})

    def fpct(v): return f"{v*100:.1f}%" if v is not None else "N/A"
    def fn(v, d=2): return f"{v:.{d}f}" if v is not None else "N/A"

    # Equity curve para JS
    eq_json = result.equity_curve[["strategy", "benchmark"]].reset_index()
    eq_json.columns = ["date", "strategy", "benchmark"]
    eq_data = eq_json.to_dict(orient="records")

    ai_status = "—"
    if agent_report:
        ai_status = f"{agent_report.audit.get('status','N/A')} | US${agent_report.audit.get('estimated_cost_usd',0):.4f}"

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QuantAI Unified 2026</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background:#f5f7fa; margin:0; padding:20px; color:#333; }}
  h1 {{ color:#1a3a5c; border-bottom: 3px solid #1f77b4; padding-bottom:10px; }}
  .metrics-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:15px; margin:20px 0; }}
  .metric-card {{ background:white; border-radius:8px; padding:15px; box-shadow:0 2px 8px rgba(0,0,0,0.1); text-align:center; }}
  .metric-card .value {{ font-size:1.8em; font-weight:bold; color:#1f77b4; }}
  .metric-card .label {{ font-size:0.85em; color:#666; margin-top:5px; }}
  .metric-card .bench {{ font-size:0.75em; color:#999; }}
  #chart {{ background:white; border-radius:8px; padding:15px; box-shadow:0 2px 8px rgba(0,0,0,0.1); margin:20px 0; }}
  .ai-box {{ background:#fff3cd; border:1px solid #ffc107; border-radius:8px; padding:15px; margin:20px 0; }}
  .footer {{ color:#999; font-size:0.8em; margin-top:30px; padding-top:15px; border-top:1px solid #ddd; }}
  table {{ width:100%; border-collapse:collapse; background:white; border-radius:8px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.1); }}
  th {{ background:#1a3a5c; color:white; padding:10px; }}
  td {{ padding:8px 10px; border-bottom:1px solid #eee; }}
  tr:nth-child(even) {{ background:#f8f9fa; }}
</style>
</head>
<body>
<h1>⚡ QuantAI Unified 2026</h1>
<p><strong>Período:</strong> {bt.get('start_date')} → {bt.get('end_date')} | 
   <strong>Capital:</strong> R$ {bt.get('initial_capital',100000):,.0f} |
   <strong>Gerado:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>

<div class="metrics-grid">
  <div class="metric-card">
    <div class="value">{fpct(m.get('cagr'))}</div>
    <div class="label">CAGR Anual</div>
    <div class="bench">Benchmark: {fpct(bm.get('cagr'))}</div>
  </div>
  <div class="metric-card">
    <div class="value">{fn(m.get('sharpe'))}</div>
    <div class="label">Sharpe Ratio</div>
    <div class="bench">Benchmark: {fn(bm.get('sharpe'))}</div>
  </div>
  <div class="metric-card">
    <div class="value">{fn(m.get('sortino'))}</div>
    <div class="label">Sortino Ratio</div>
    <div class="bench">Meta: > 1.5</div>
  </div>
  <div class="metric-card">
    <div class="value" style="color:#d62728">{fpct(m.get('max_drawdown'))}</div>
    <div class="label">Max Drawdown</div>
    <div class="bench">Benchmark: {fpct(bm.get('max_drawdown'))}</div>
  </div>
  <div class="metric-card">
    <div class="value">{fn(m.get('calmar'))}</div>
    <div class="label">Calmar Ratio</div>
    <div class="bench">Meta: > 0.5</div>
  </div>
  <div class="metric-card">
    <div class="value">R$ {m.get('final_equity',0):,.0f}</div>
    <div class="label">Capital Final</div>
    <div class="bench">Win Rate: {fpct(m.get('win_rate'))}</div>
  </div>
</div>

<div id="chart"></div>

<div class="ai-box">
  <strong>🤖 Agente IA:</strong> {ai_status}
</div>

<script>
const data = {json.dumps(eq_data, default=str)};
const dates = data.map(d => d.date);
const strategy = data.map(d => d.strategy);
const benchmark = data.map(d => d.benchmark);

Plotly.newPlot('chart', [
  {{x: dates, y: strategy, name: 'Estratégia', line: {{color:'#1f77b4', width:2}}}},
  {{x: dates, y: benchmark, name: 'Benchmark (^BVSP)', line: {{color:'#ff7f0e', width:1.5, dash:'dash'}}}}
], {{
  title: 'Curva de Capital — QuantAI Unified 2026',
  yaxis: {{title: 'Capital (R$)', tickformat: 'R$,.0f'}},
  xaxis: {{title: 'Data'}},
  plot_bgcolor: '#ffffff',
  paper_bgcolor: '#ffffff',
  legend: {{orientation: 'h', y: -0.15}}
}}, {{responsive: true}});
</script>

<p class="footer">
⚠️ Aviso Legal: Este relatório é educacional. Resultados de backtest não garantem performance futura. 
Nenhuma informação constitui recomendação de investimento. | QuantAI Unified v2.0.0
</p>
</body>
</html>"""

    html_path = reports_dir / "dashboard.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path