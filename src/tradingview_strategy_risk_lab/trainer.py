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
        win_rate=round(float((frame["pnl_r"] > 0).mean()), 4),
        profit_factor=_profit_factor(frame["pnl_r"]),
        average_r=round(float(frame["pnl_r"].mean()), 4),
        max_drawdown_r=_max_drawdown(frame["pnl_r"]),
        roc_auc=round(float(auc), 4),
        holdout_rows=len(x_test),
        feature_count=len(FEATURE_COLUMNS),
        evidence_checks=_evidence_checks(frame, config, len(x_test)),
        top_features=top_features,
        filter_suggestions=_filter_suggestions(frame),
        counterfactual_hint=_counterfactual_hint(top_features),
    )


def _feature_importance(importances: np.ndarray) -> list[dict[str, float | str]]:
    order = np.argsort(importances)[::-1][:5]
    return [
        {"feature": FEATURE_COLUMNS[index], "importance": round(float(importances[index]), 5)}
        for index in order
    ]


def _evidence_checks(
    frame: pd.DataFrame,
    config: TrainingConfig,
    holdout_rows: int,
) -> list[dict[str, str]]:
    forbidden_features = {"pnl_r", "exit_price", "trade_id"}
    leakage_features = sorted(forbidden_features.intersection(FEATURE_COLUMNS))
    class_counts = (frame["pnl_r"] <= config.bad_trade_threshold_r).astype(int).value_counts()
    return [
        {
            "check": "holdout_split",
            "status": "pass",
            "evidence": f"{holdout_rows} trades reserved for holdout scoring.",
        },
        {
            "check": "zero_lookahead_features",
            "status": "pass" if not leakage_features else "fail",
            "evidence": "No post-exit fields are used as features."
            if not leakage_features
            else f"Remove leakage features: {', '.join(leakage_features)}.",
        },
        {
            "check": "minimum_class_count",
            "status": "pass" if int(class_counts.min()) >= 5 else "review",
            "evidence": f"Bad/good trade class counts: {class_counts.to_dict()}.",
        },
        {
            "check": "claim_boundary",
            "status": "review",
            "evidence": (
                "Risk model explains historical fragility; it is not a live execution rule."
            ),
        },
    ]


def _counterfactual_hint(top_features: list[dict[str, float | str]]) -> str:
    if not top_features:
        return "No stable driver found."
    feature = str(top_features[0]["feature"])
    return (
        f"Start the counterfactual review with {feature}. Compare the bad-trade rate before "
        "and after tightening that condition before using it as a live filter."
    )


def _profit_factor(pnl_r: pd.Series) -> float:
    gains = float(pnl_r[pnl_r > 0].sum())
    losses = abs(float(pnl_r[pnl_r < 0].sum()))
    if losses == 0:
        return round(gains, 4)
    return round(gains / losses, 4)


def _max_drawdown(pnl_r: pd.Series) -> float:
    equity = pnl_r.cumsum()
    drawdown = equity - equity.cummax()
    return round(abs(float(drawdown.min())), 4)


def _filter_suggestions(frame: pd.DataFrame) -> list[dict[str, float | str]]:
    suggestions: list[dict[str, float | str]] = []
    bad_trade = frame["pnl_r"] <= 0
    candidates = [
        ("atr_pct", ">=", float(frame["atr_pct"].quantile(0.75))),
        ("volume_z", "<=", float(frame["volume_z"].quantile(0.25))),
        ("trend_score", "<=", float(frame["trend_score"].quantile(0.25))),
    ]
    overall_bad_rate = float(bad_trade.mean())
    for feature, operator, threshold in candidates:
        mask = frame[feature] >= threshold if operator == ">=" else frame[feature] <= threshold
        if int(mask.sum()) < 3:
            continue
        condition_bad_rate = float(bad_trade[mask].mean())
        if condition_bad_rate <= overall_bad_rate:
            continue
        suggestions.append(
            {
                "condition": f"{feature} {operator} {threshold:.4f}",
                "bad_trade_rate": round(condition_bad_rate, 4),
                "baseline_bad_trade_rate": round(overall_bad_rate, 4),
                "suggested_filter": (
                    f"Review trades where {feature} is {operator} {threshold:.4f}; "
                    "do not deploy the filter until it survives holdout review."
                ),
            }
        )
    return suggestions[:3]
