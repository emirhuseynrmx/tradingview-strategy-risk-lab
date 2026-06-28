from tradingview_strategy_risk_lab.data import FEATURE_COLUMNS


def test_feature_columns_do_not_include_outcome_or_future_fields() -> None:
    forbidden = {
        "exit_price",
        "pnl_r",
        "entry_return",
        "realized_return",
        "max_favorable_excursion",
        "max_adverse_excursion",
    }

    assert forbidden.isdisjoint(FEATURE_COLUMNS)
