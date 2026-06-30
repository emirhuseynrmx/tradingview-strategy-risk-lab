from __future__ import annotations

from enum import Enum
from pathlib import Path

import typer

from tradingview_strategy_risk_lab.data import load_trades
from tradingview_strategy_risk_lab.explainability import format_driver_block
from tradingview_strategy_risk_lab.schemas import TrainingConfig
from tradingview_strategy_risk_lab.trainer import train_strategy_risk_model

app = typer.Typer(help="Train a strategy-risk model from TradingView trade exports.")


class OutputFormat(str, Enum):
    summary = "summary"
    json = "json"


@app.command()
def train(
    csv_path: Path = Path("examples/trades.csv"),
    output: OutputFormat = OutputFormat.summary,
    bad_trade_threshold: float = 0.0,
) -> None:
    """Train a risk model and display results."""
    config = TrainingConfig(min_rows=10, bad_trade_threshold_r=bad_trade_threshold)
    report = train_strategy_risk_model(load_trades(csv_path), config)

    if output == OutputFormat.json:
        typer.echo(report.model_dump_json(indent=2))
        return

    # Human-readable summary
    typer.echo("")
    typer.echo("TradingView Strategy Risk Lab")
    typer.echo("=" * 45)
    typer.echo(f"  Trades analysed : {report.rows}")
    typer.echo(f"  Holdout rows    : {report.holdout_rows}")
    typer.echo(f"  Win rate        : {report.win_rate:.1%}")
    typer.echo(f"  Bad-trade rate  : {report.bad_trade_rate:.1%}")
    typer.echo(f"  Profit factor   : {report.profit_factor:.3f}")
    typer.echo(f"  Average R       : {report.average_r:+.3f}")
    typer.echo(f"  Max drawdown R  : {report.max_drawdown_r:.3f}")
    typer.echo(f"  ROC-AUC         : {report.roc_auc:.4f}")

    typer.echo("")
    typer.echo("Risk Drivers")
    typer.echo("-" * 45)
    drivers = report.shap_features if report.shap_features else report.top_features
    use_shap = bool(report.shap_features)
    typer.echo(format_driver_block(drivers, use_shap=use_shap))

    if report.filter_suggestions:
        typer.echo("")
        typer.echo("Fragile Setup Signals")
        typer.echo("-" * 45)
        for s in report.filter_suggestions:
            typer.echo(f"  {s['condition']}")
            typer.echo(
                f"    bad-trade rate {s['bad_trade_rate']:.1%}  "
                f"vs baseline {s['baseline_bad_trade_rate']:.1%}"
            )

    if report.counterfactual_suggestions:
        typer.echo("")
        typer.echo("Counterfactual Next Steps")
        typer.echo("-" * 45)
        for i, step in enumerate(report.counterfactual_suggestions, 1):
            typer.echo(f"  {i}. {step}")

    typer.echo("")
    typer.echo("Evidence Contract")
    typer.echo("-" * 45)
    for check in report.evidence_checks:
        icon = "OK" if check.status == "pass" else "!!" if check.status == "review" else "--"
        typer.echo(f"  [{icon}] {check.check}: {check.evidence}")
    typer.echo("")
