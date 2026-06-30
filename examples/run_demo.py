"""
End-to-end demo: train the strategy risk model on the sample trade export and print a full report.

Usage:
    python examples/run_demo.py
    python examples/run_demo.py --csv examples/trades.csv --bad-threshold -0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tradingview_strategy_risk_lab.data import load_trades
from tradingview_strategy_risk_lab.explainability import format_driver_block
from tradingview_strategy_risk_lab.schemas import TrainingConfig
from tradingview_strategy_risk_lab.trainer import train_strategy_risk_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("examples/trades.csv"))
    parser.add_argument(
        "--bad-threshold",
        type=float,
        default=0.0,
        help="Trades at or below this R-multiple are labelled bad (default 0.0 = any loss)",
    )
    args = parser.parse_args()

    print(f"\nLoading trade export: {args.csv}")
    frame = load_trades(args.csv)
    config = TrainingConfig(min_rows=10, bad_trade_threshold_r=args.bad_threshold)

    print("Training RandomForest risk model + computing SHAP values…\n")
    report = train_strategy_risk_model(frame, config)

    print("TradingView Strategy Risk Lab — Full Report")
    print("=" * 55)
    print(f"  Trades          : {report.rows}")
    print(f"  Win rate        : {report.win_rate:.1%}")
    print(f"  Bad-trade rate  : {report.bad_trade_rate:.1%}")
    print(f"  Profit factor   : {report.profit_factor:.3f}")
    print(f"  Average R       : {report.average_r:+.4f}")
    print(f"  Max drawdown R  : {report.max_drawdown_r:.4f}")
    print(f"  ROC-AUC         : {report.roc_auc:.4f}  (holdout n={report.holdout_rows})")

    print("\nRF Feature Importances")
    print("-" * 45)
    print(format_driver_block(report.top_features, use_shap=False))

    if report.shap_features:
        print("\nSHAP Feature Importances (TreeExplainer, |mean|)")
        print("-" * 45)
        print(format_driver_block(report.shap_features, use_shap=True))

    if report.filter_suggestions:
        print("\nFragile Setup Conditions")
        print("-" * 45)
        for s in report.filter_suggestions:
            print(f"  Condition  : {s['condition']}")
            print(
                f"  Bad rate   : {s['bad_trade_rate']:.1%} "
                f"vs baseline {s['baseline_bad_trade_rate']:.1%}"
            )
            print(f"  Suggestion : {s['suggested_filter']}")
            print()

    if report.counterfactual_suggestions:
        print("Counterfactual Next Steps")
        print("-" * 45)
        for i, step in enumerate(report.counterfactual_suggestions, 1):
            print(f"  {i}. {step}")

    print("\nEvidence Contract")
    print("-" * 45)
    for check in report.evidence_checks:
        icon = "OK" if check.status == "pass" else "!!" if check.status == "review" else "--"
        print(f"  [{icon}] {check.check}: {check.evidence}")
    print()


if __name__ == "__main__":
    main()
