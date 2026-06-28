from __future__ import annotations

from litestar import Litestar, get, post

from tradingview_strategy_risk_lab.data import load_trades
from tradingview_strategy_risk_lab.explainability import shap_and_dice_stack_note
from tradingview_strategy_risk_lab.schemas import StrategyRiskReport, TrainRequest
from tradingview_strategy_risk_lab.trainer import train_strategy_risk_model


@get("/health", sync_to_thread=False)
def health() -> dict[str, str]:
    return {"status": "ok"}


@post("/v1/train", sync_to_thread=False)
def train(data: TrainRequest) -> StrategyRiskReport:
    frame = load_trades(data.csv_path)
    return train_strategy_risk_model(frame, data.config)


@get("/v1/explainability-stack", sync_to_thread=False)
def explainability_stack() -> dict[str, str]:
    return shap_and_dice_stack_note()


app = Litestar(route_handlers=[health, train, explainability_stack], debug=False)
