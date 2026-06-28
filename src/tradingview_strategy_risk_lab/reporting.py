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
    return f"""#set page(margin: 15mm)
#set text(font: "Arial", size: 10pt)
#set heading(numbering: none)

#let ink = rgb("#111827")
#let muted = rgb("#64748b")
#let panel = rgb("#f3f6fb")
#let line = rgb("#d7dee8")
#let risk = rgb("#b91c1c")

#let card(label, value, note: none) = block[
  #rect(fill: panel, radius: 6pt, inset: 9pt, width: 100%)[
    #text(size: 8pt, fill: muted, weight: "bold")[#upper(label)]
    #linebreak()
    #text(size: 19pt, fill: ink, weight: "bold")[#value]
    #if note != none [#linebreak() #text(size: 8pt, fill: muted)[#note]]
  ]
]

#let check(label, value) = block[
  #text(weight: "bold")[#label]
  #linebreak()
  #text(fill: muted)[#value]
]

#text(size: 22pt, weight: "bold", fill: ink)[TradingView Strategy Risk Lab]

#text(fill: muted)[
  Model-training report for a TradingView trade export. The report reviews fragile
  setups with holdout metrics, SHAP drivers, and a zero-lookahead feature boundary.
]

#grid(columns: (1fr, 1fr, 1fr), gutter: 8pt)[
  #card("Trades", "{report.rows}", note: "validated rows")
][
  #card("Bad-trade rate", "{report.bad_trade_rate}", note: "label prevalence")
][
  #card("ROC-AUC", "{report.roc_auc}", note: "holdout score")
]

#v(10pt)
#grid(columns: (1.45fr, 1fr), gutter: 14pt)[
  #text(size: 13pt, weight: "bold")[Top Risk Drivers]
  #v(4pt)
  #table(
    columns: (1fr, .7fr),
    stroke: line,
    inset: 5pt,
    [*Feature*], [*Importance*],
{feature_rows}
  )
][
  #text(size: 13pt, weight: "bold")[Evidence Contract]
  #v(4pt)
  #block(stroke: line, radius: 6pt, inset: 8pt)[
    #check("Feature timing", "Only setup fields known before exit are model inputs.")
    #v(5pt)
    #check("Label boundary", "Realized PnL is used only to define the target.")
    #v(5pt)
    #check("Explainability", "SHAP shows drivers; DiCE stays on actionable fields.")
    #v(5pt)
    #check("Usage", "This is strategy review, not live execution.")
  ]
]

#v(10pt)
#block(fill: rgb("#fef2f2"), radius: 6pt, inset: 8pt)[
  #text(weight: "bold", fill: risk)[Counterfactual next step:] {report.counterfactual_hint}
]
"""

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


if __name__ == "__main__":
    main()
