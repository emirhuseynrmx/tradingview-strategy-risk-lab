from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TradeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    trade_id: str = Field(min_length=1, max_length=64)
    side: str = Field(pattern="^(long|short)$")
    entry_price: float = Field(gt=0)
    exit_price: float = Field(gt=0)
    pnl_r: float = Field(ge=-50, le=50)
    bars_held: int = Field(ge=1, le=10_000)
    atr_pct: float = Field(ge=0, le=1)
    volume_z: float = Field(ge=-10, le=10)
    trend_score: float = Field(ge=-5, le=5)
    session: str = Field(min_length=2, max_length=32)

    @field_validator("side", "session")
    @classmethod
    def lower(cls, value: str) -> str:
        return value.lower()


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bad_trade_threshold_r: float = Field(default=0.0, ge=-5, le=5)
    min_rows: int = Field(default=30, ge=10)
    test_size: float = Field(default=0.25, gt=0, lt=0.5)
    random_state: int = Field(default=42, ge=0)


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    csv_path: Path
    config: TrainingConfig = Field(default_factory=TrainingConfig)


class StrategyRiskReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: int
    bad_trade_rate: float
    roc_auc: float
    top_features: list[dict[str, float | str]]
    counterfactual_hint: str

