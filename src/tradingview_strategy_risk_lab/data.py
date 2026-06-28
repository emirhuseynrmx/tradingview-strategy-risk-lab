from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera import Check

from tradingview_strategy_risk_lab.schemas import TradeRecord

FEATURE_COLUMNS = ["side_code", "atr_pct", "volume_z", "trend_score"]

TRADE_FRAME_SCHEMA = pa.DataFrameSchema(
    {
        "trade_id": pa.Column(str, unique=True),
        "side": pa.Column(str, Check.isin(["long", "short"])),
        "entry_price": pa.Column(float, Check.gt(0)),
        "exit_price": pa.Column(float, Check.gt(0)),
        "pnl_r": pa.Column(float, Check.in_range(-50, 50)),
        "bars_held": pa.Column(int, Check.in_range(1, 10_000)),
        "atr_pct": pa.Column(float, Check.in_range(0, 1)),
        "volume_z": pa.Column(float, Check.in_range(-10, 10)),
        "trend_score": pa.Column(float, Check.in_range(-5, 5)),
        "session": pa.Column(str),
        "side_code": pa.Column(float, Check.isin([-1.0, 1.0])),
    },
    strict=True,
)


def load_trades(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    records = [TradeRecord.model_validate(row) for row in raw.to_dict(orient="records")]
    frame = pd.DataFrame([record.model_dump() for record in records])
    frame["side_code"] = frame["side"].map({"long": 1.0, "short": -1.0})
    return TRADE_FRAME_SCHEMA.validate(frame)
