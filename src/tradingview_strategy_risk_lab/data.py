from __future__ import annotations

from pathlib import Path

import pandas as pd

from tradingview_strategy_risk_lab.schemas import TradeRecord

FEATURE_COLUMNS = ["side_code", "entry_return", "bars_held", "atr_pct", "volume_z", "trend_score"]


def load_trades(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    records = [TradeRecord.model_validate(row) for row in raw.to_dict(orient="records")]
    frame = pd.DataFrame([record.model_dump() for record in records])
    frame["side_code"] = frame["side"].map({"long": 1.0, "short": -1.0})
    frame["entry_return"] = (frame["exit_price"] - frame["entry_price"]) / frame["entry_price"]
    return frame

