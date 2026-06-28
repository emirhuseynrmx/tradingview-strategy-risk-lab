from tradingview_strategy_risk_lab.reporting import build_report_typ


def test_report_contains_model_metrics() -> None:
    report = build_report_typ()

    assert "TradingView Strategy Risk Lab" in report
    assert "ROC-AUC" in report
    assert "Top Risk Drivers" in report
