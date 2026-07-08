"""QuantAI Unified — agent.py
Agente OpenAI auditável com fallback determinístico.
Baseado no Gabriel18182/QuantAI com melhorias de prompt e auditoria.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

AUDIT_TEMPLATE = {
    "provider": "openai",
    "model": None,
    "used_live_agent": False,
    "agent_sdk_used": False,
    "estimated_cost_usd": 0.0,
    "tokens_used": 0,
    "status": "not_generated",
    "timestamp": None,
    "fallback_reason": None,
}


@dataclass
class AgentReport:
    """Parecer gerado pelo agente IA."""
    summary: str
    risk_analysis: str
    recommendations: str
    market_outlook: str
    audit: Dict[str, Any]


def generate_agent_report(
    metrics: Dict,
    benchmark_metrics: Dict,
    weights: Any,
    config: Dict,
    sector_exposure: Optional[Dict] = None,
) -> AgentReport:
    """
    Gera parecer executivo via agente OpenAI.
    Fallback determinístico auditado se API indisponível.
    """
    ai_cfg = config.get("ai", {})
    model = os.environ.get("OPENAI_MODEL", ai_cfg.get("model", "gpt-4o-mini"))
    api_key = os.environ.get("OPENAI_API_KEY", "")

    audit = AUDIT_TEMPLATE.copy()
    audit["model"] = model
    audit["timestamp"] = datetime.now().isoformat()

    if not api_key or api_key.startswith("sk-placeholder"):
        logger.info("Chave OpenAI não configurada. Usando fallback auditado.")
        audit["status"] = "fallback_no_key"
        audit["fallback_reason"] = "OPENAI_API_KEY não configurada"
        return _build_fallback_report(metrics, benchmark_metrics, weights, config, audit, sector_exposure)

    # Tenta OpenAI Agents SDK
    try:
        report = _call_openai_agents_sdk(
            metrics, benchmark_metrics, weights, config, model, api_key, audit, sector_exposure
        )
        return report
    except ImportError:
        logger.warning("OpenAI Agents SDK não disponível. Tentando client direto.")
    except Exception as exc:
        logger.warning("Agents SDK falhou (%s). Tentando client direto.", exc)

    # Tenta client OpenAI direto
    try:
        report = _call_openai_direct(
            metrics, benchmark_metrics, weights, config, model, api_key, audit, sector_exposure
        )
        return report
    except Exception as exc:
        logger.warning("OpenAI direto falhou (%s). Usando fallback auditado.", exc)
        audit["status"] = f"fallback_api_error: {str(exc)[:100]}"
        audit["fallback_reason"] = str(exc)
        return _build_fallback_report(metrics, benchmark_metrics, weights, config, audit, sector_exposure)


def _call_openai_agents_sdk(
    metrics, benchmark_metrics, weights, config, model, api_key, audit, sector_exposure
) -> AgentReport:
    """Chama OpenAI via Agents SDK."""
    from agents import Agent, Runner
    import os
    os.environ["OPENAI_API_KEY"] = api_key

    prompt = _build_prompt(metrics, benchmark_metrics, weights, config, sector_exposure)

    agent = Agent(
        name="QuantAI-Comite-Risco",
        instructions=(
            "Você é um comitê de análise quantitativa especializado em mercado brasileiro (B3). "
            "Analise os resultados do backtest e forneça parecer executivo estruturado em 4 seções: "
            "1) Resumo Executivo, 2) Análise de Riscos, 3) Recomendações, 4) Perspectiva de Mercado. "
            "Use linguagem profissional, cite as métricas relevantes e compare com benchmarks. "
            "Responda em português do Brasil."
        ),
        model=model,
    )

    t0 = time.time()
    result = Runner.run_sync(agent, prompt)
    elapsed = time.time() - t0

    text = result.final_output or ""
    sections = _parse_sections(text)

    # Estimativa de custo (gpt-4o-mini: ~$0.15/1M input, $0.60/1M output)
    tokens_est = len(prompt.split()) * 1.3 + len(text.split()) * 1.3
    cost_est = tokens_est * 0.0000003  # ~US$0.30/1M tokens (média input+output)

    audit["used_live_agent"] = True
    audit["agent_sdk_used"] = True
    audit["estimated_cost_usd"] = round(cost_est, 6)
    audit["tokens_used"] = int(tokens_est)
    audit["status"] = "live_agents_sdk"
    audit["elapsed_seconds"] = round(elapsed, 2)

    logger.info("AgentSDK: %.2fs | ~%d tokens | ~US$%.4f", elapsed, tokens_est, cost_est)
    return AgentReport(**sections, audit=audit)


def _call_openai_direct(
    metrics, benchmark_metrics, weights, config, model, api_key, audit, sector_exposure
) -> AgentReport:
    """Chama OpenAI via client direto (sem Agents SDK)."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    prompt = _build_prompt(metrics, benchmark_metrics, weights, config, sector_exposure)

    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um comitê de análise quantitativa especializado em mercado brasileiro (B3). "
                    "Analise os resultados do backtest e forneça parecer executivo estruturado. "
                    "Responda em português do Brasil com linguagem profissional."
                )
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.3,
    )
    elapsed = time.time() - t0
    text = response.choices[0].message.content or ""
    sections = _parse_sections(text)

    usage = response.usage
    tokens_est = (usage.prompt_tokens + usage.completion_tokens) if usage else 1000
    cost_est = tokens_est * 0.0000003

    audit["used_live_agent"] = True
    audit["agent_sdk_used"] = False
    audit["estimated_cost_usd"] = round(cost_est, 6)
    audit["tokens_used"] = int(tokens_est)
    audit["status"] = "live_direct_api"
    audit["elapsed_seconds"] = round(elapsed, 2)

    logger.info("OpenAI direto: %.2fs | %d tokens | ~US$%.4f", elapsed, tokens_est, cost_est)
    return AgentReport(**sections, audit=audit)


def _build_prompt(metrics, benchmark_metrics, weights, config, sector_exposure) -> str:
    """Constrói o prompt de análise para o agente."""
    def fmt_pct(v): return f"{v*100:.1f}%" if v is not None else "N/A"
    def fmt_n(v, d=2): return f"{v:.{d}f}" if v is not None else "N/A"

    top_assets = ""
    if hasattr(weights, "items"):
        sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_assets = ", ".join(f"{a} ({w*100:.1f}%)" for a, w in sorted_w[:5])
    elif hasattr(weights, "iloc") and len(weights) > 0:
        last = weights.iloc[-1].sort_values(ascending=False)
        top_assets = ", ".join(f"{a} ({w*100:.1f}%)" for a, w in last.head(5).items())

    sector_str = ""
    if sector_exposure:
        sector_str = "Exposição setorial: " + ", ".join(
            f"{s}: {w*100:.1f}%" for s, w in list(sector_exposure.items())[:5]
        )

    bt = config.get("backtest", {})

    return f"""
ANÁLISE DE BACKTEST — QuantAI Unified 2026
Período: {bt.get('start_date')} a {bt.get('end_date')}
Capital inicial: R$ {bt.get('initial_capital', 100000):,.0f}

MÉTRICAS DA ESTRATÉGIA:
- CAGR: {fmt_pct(metrics.get('cagr'))} (Benchmark: {fmt_pct(benchmark_metrics.get('cagr'))})
- Retorno Total: {fmt_pct(metrics.get('total_return'))}
- Sharpe Ratio: {fmt_n(metrics.get('sharpe'))} (Benchmark: {fmt_n(benchmark_metrics.get('sharpe'))})
- Sortino Ratio: {fmt_n(metrics.get('sortino'))}
- Max Drawdown: {fmt_pct(metrics.get('max_drawdown'))} (Benchmark: {fmt_pct(benchmark_metrics.get('max_drawdown'))})
- Calmar Ratio: {fmt_n(metrics.get('calmar'))}
- Volatilidade: {fmt_pct(metrics.get('volatility'))} (Benchmark: {fmt_pct(benchmark_metrics.get('volatility'))})
- VaR 95% (1d): {fmt_pct(metrics.get('var_95_daily'))}
- Win Rate: {fmt_pct(metrics.get('win_rate'))}
- Capital Final: R$ {metrics.get('final_equity', 0):,.0f}

CARTEIRA ATUAL (Top 5): {top_assets}
{sector_str}

Forneça parecer executivo completo com:
1. **Resumo Executivo** — avaliação geral da performance vs benchmark
2. **Análise de Riscos** — principais riscos identificados (drawdown, concentração, volatilidade)
3. **Recomendações** — melhorias sugeridas para a estratégia
4. **Perspectiva de Mercado** — contexto macroeconômico brasileiro relevante
"""


def _parse_sections(text: str) -> Dict:
    """Extrai seções do texto gerado pelo agente."""
    import re

    def extract(pattern, fallback=""):
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else fallback

    summary = extract(r"resumo executivo[:\*\n]+(.*?)(?=análise de riscos|2\.|$)", text[:500])
    risks   = extract(r"análise de riscos[:\*\n]+(.*?)(?=recomendações|3\.|$)", "Ver relatório completo.")
    recs    = extract(r"recomendações[:\*\n]+(.*?)(?=perspectiva|4\.|$)", "Ver relatório completo.")
    outlook = extract(r"perspectiva[:\*\n]+(.*?)$", "Ver relatório completo.")

    if not summary:
        summary = text[:600] + "..." if len(text) > 600 else text

    return {
        "summary":        summary or text[:400],
        "risk_analysis":  risks,
        "recommendations": recs,
        "market_outlook":  outlook,
    }


def _build_fallback_report(
    metrics, benchmark_metrics, weights, config, audit, sector_exposure
) -> AgentReport:
    """
    Fallback determinístico: gera parecer baseado em regras fixas.
    Auditável e reproduzível sem chave OpenAI.
    """
    cagr  = metrics.get("cagr", 0) * 100
    sharpe = metrics.get("sharpe", 0)
    mdd   = metrics.get("max_drawdown", 0) * 100
    bench_cagr = benchmark_metrics.get("cagr", 0) * 100
    alpha = cagr - bench_cagr

    # Avaliação baseada em regras
    if sharpe > 1.0 and alpha > 3:
        perf_class = "EXCELENTE"
        perf_desc = "significativamente superior ao benchmark"
    elif sharpe > 0.5 and alpha > 0:
        perf_class = "BOA"
        perf_desc = "superior ao benchmark com risco controlado"
    elif alpha > 0:
        perf_class = "ADEQUADA"
        perf_desc = "levemente superior ao benchmark"
    else:
        perf_class = "ABAIXO DO ESPERADO"
        perf_desc = "inferior ao benchmark — revisão necessária"

    risk_level = "ALTO" if mdd < -25 else "MODERADO" if mdd < -15 else "BAIXO"

    audit["status"] = "fallback_deterministic"

    return AgentReport(
        summary=(
            f"A estratégia QuantAI registrou performance {perf_class}: CAGR de {cagr:.1f}% "
            f"vs {bench_cagr:.1f}% do benchmark, gerando alpha de {alpha:.1f} p.p. ao ano. "
            f"O Sharpe Ratio de {sharpe:.2f} indica retorno {perf_desc}. "
            f"[FALLBACK AUDITADO — execute com OPENAI_API_KEY para análise generativa]"
        ),
        risk_analysis=(
            f"Risco classificado como {risk_level}: Maximum Drawdown de {mdd:.1f}%. "
            f"{'Drawdown elevado sugere exposição a regime de queda. Considere filtro de regime mais agressivo.' if risk_level == 'ALTO' else ''}"
            f"Volatilidade anualizada: {metrics.get('volatility', 0)*100:.1f}% "
            f"({'acima' if metrics.get('volatility', 0) > 0.20 else 'dentro'} da meta de 20% a.a.)."
        ),
        recommendations=(
            "1. Considere aumentar o filtro de liquidez mínima para R$2M/dia. "
            "2. Avalie inclusão de FIIs para reduzir correlação com renda variável. "
            "3. Implemente stop-loss dinâmico baseado em ATR para proteção de drawdown. "
            "4. Configure OPENAI_API_KEY para análise generativa real com insights de sentimento."
        ),
        market_outlook=(
            "Mercado brasileiro em 2024-2025 sob influência de: ciclo de juros (Selic), "
            "risco fiscal, câmbio e commodities. Monitor macro BCB/SGS ativo. "
            "Selic elevada aumenta custo de oportunidade vs renda fixa — exige Sharpe > 1.0 "
            "para justificar alocação em renda variável no contexto brasileiro."
        ),
        audit=audit,
    )