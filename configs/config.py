"""QuantAI Unified — config.py
Carregamento e validação de configuração YAML.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class QuantAIConfig:
    raw: Dict[str, Any]
    config_path: Path

    @property
    def reports_dir(self) -> Path:
        d = self.config_path.parent.parent / "reports"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def data_dir(self) -> Path:
        d = self.config_path.parent.parent / "data" / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def assets(self) -> List[str]:
        return self.raw.get("assets", [])

    @property
    def start_date(self) -> str:
        return self.raw["backtest"]["start_date"]

    @property
    def end_date(self) -> str:
        return self.raw["backtest"]["end_date"]

    @property
    def initial_capital(self) -> float:
        return float(self.raw["backtest"]["initial_capital"])

    @property
    def transaction_cost_bps(self) -> float:
        return float(self.raw["backtest"].get("transaction_cost_bps", 10))

    @property
    def slippage_bps(self) -> float:
        return float(self.raw["backtest"].get("slippage_bps", 5))

    @property
    def benchmark(self) -> str:
        return self.raw["backtest"].get("benchmark", "^BVSP")

    @property
    def top_n(self) -> int:
        return int(self.raw["selection"]["top_n"])

    @property
    def max_weight(self) -> float:
        return float(self.raw["selection"]["max_weight"])

    @property
    def min_weight(self) -> float:
        return float(self.raw["selection"].get("min_weight", 0.02))

    @property
    def ai_model(self) -> str:
        return os.environ.get("OPENAI_MODEL", self.raw.get("ai", {}).get("model", "gpt-4o-mini"))

    @property
    def sector_map(self) -> Dict[str, str]:
        return self.raw.get("sector_map", {})


def load_config(config_path: str | Path) -> QuantAIConfig:
    """Carrega e valida o arquivo de configuração YAML."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config não encontrado: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return QuantAIConfig(raw=raw, config_path=config_path)