from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from tradingview_strategy_risk_lab.data import load_trades
from tradingview_strategy_risk_lab.schemas import TrainingConfig
from tradingview_strategy_risk_lab.trainer import train_strategy_risk_model


def build_report_typ(csv_path: Path = Path("examples/trades.csv")) -> str:
    report = train_strategy_risk_model(load_trades(csv_path), TrainingConfig(min_rows=10))
    feature_rows = "\n".join(
        f'  [{item["feature"]}], [{item["importance"]}],'
        for item in report.top_features
    )
    rows_card = _metric_card("Trades", str(report.rows))
    bad_rate_card = _metric_card("Bad-trade rate", str(report.bad_trade_rate))
    auc_card = _metric_card("ROC-AUC", str(report.roc_auc))
    return f"""#set page(margin: 16mm)
#set text(font: "Arial", size: 10pt)

#text(size: 18pt, weight: "bold")[TradingView Strategy Risk Lab]

Model-training report for a TradingView trade export.

#grid(columns: (1fr, 1fr, 1fr), gutter: 8pt)[
{rows_card}
][
{bad_rate_card}
][
{auc_card}
]

#v(10pt)
#text(size: 12pt, weight: "bold")[Top Risk Drivers]

#table(columns: (1fr, 1fr), [Feature], [Importance],
{feature_rows}
)

#v(8pt)
#text(weight: "bold")[Counterfactual next step:] {report.counterfactual_hint}
"""


def _metric_card(label: str, value: str) -> str:
    return (
        f'  #block(fill: rgb("#f3f6fb"), radius: 4pt, inset: 8pt)'
        f'[{label}\\ #text(size: 18pt, weight: "bold")[{value}]]'
    )


def write_report(output_dir: Path = Path("docs/samples")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    typ_path = output_dir / "strategy_risk_report.typ"
    typ_path.write_text(build_report_typ(), encoding="utf-8")
    if shutil.which("typst"):
        subprocess.run(
            ["typst", "compile", str(typ_path), str(typ_path.with_suffix(".pdf"))],
            check=True,
        )
    return typ_path


def main() -> None:
    print(write_report())
