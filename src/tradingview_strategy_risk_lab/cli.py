from __future__ import annotations

from pathlib import Path

import typer

from tradingview_strategy_risk_lab.data import load_trades
from tradingview_strategy_risk_lab.schemas import TrainingConfig
from tradingview_strategy_risk_lab.trainer import train_strategy_risk_model

app = typer.Typer(help="Train a strategy-risk model from TradingView trade exports.")


@app.command()
def train(csv_path: Path = Path("examples/trades.csv")) -> None:
    report = train_strategy_risk_model(load_trades(csv_path), TrainingConfig(min_rows=10))
    typer.echo(report.model_dump_json(indent=2))

