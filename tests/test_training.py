from pathlib import Path

from tradingview_strategy_risk_lab.data import load_trades
from tradingview_strategy_risk_lab.schemas import TrainingConfig
from tradingview_strategy_risk_lab.trainer import train_strategy_risk_model


def test_train_strategy_risk_model() -> None:
    frame = load_trades(Path("examples/trades.csv"))
    report = train_strategy_risk_model(frame, TrainingConfig(min_rows=10, test_size=0.3))

    assert report.rows >= 30
    assert 0 <= report.roc_auc <= 1
    assert report.top_features
    assert "counterfactual" in report.counterfactual_hint
