from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tradingview_strategy_risk_lab.data import FEATURE_COLUMNS
from tradingview_strategy_risk_lab.schemas import StrategyRiskReport, TrainingConfig


def train_strategy_risk_model(frame: pd.DataFrame, config: TrainingConfig) -> StrategyRiskReport:
    if len(frame) < config.min_rows:
        raise ValueError(f"strategy risk modeling needs at least {config.min_rows} trades")

    target = (frame["pnl_r"] <= config.bad_trade_threshold_r).astype(int)
    if target.nunique() != 2:
        raise ValueError("dataset must contain both winning and losing trades")

    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURE_COLUMNS],
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target,
    )
    model = RandomForestClassifier(
        n_estimators=120,
        min_samples_leaf=3,
        random_state=config.random_state,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)

    top_features = _feature_importance(model.feature_importances_)
    return StrategyRiskReport(
        rows=len(frame),
        bad_trade_rate=round(float(target.mean()), 4),
        roc_auc=round(float(auc), 4),
        top_features=top_features,
        counterfactual_hint=_counterfactual_hint(top_features),
    )


def _feature_importance(importances: np.ndarray) -> list[dict[str, float | str]]:
    order = np.argsort(importances)[::-1][:5]
    return [
        {"feature": FEATURE_COLUMNS[index], "importance": round(float(importances[index]), 5)}
        for index in order
    ]


def _counterfactual_hint(top_features: list[dict[str, float | str]]) -> str:
    if not top_features:
        return "No stable driver found."
    feature = str(top_features[0]["feature"])
    return f"Use DiCE to search how changing {feature} moves a trade away from high-risk."

