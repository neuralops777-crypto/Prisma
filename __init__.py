"""QuantAI Unified — pacote principal."""
from configs.config import load_config
from data import load_prices
from backtest_engine.backtest import run_backtest
from reports.reporting import save_outputs, build_markdown_report

__version__ = "2.0.0"
__all__ = ["load_config", "load_prices", "run_backtest", "save_outputs", "build_markdown_report"]