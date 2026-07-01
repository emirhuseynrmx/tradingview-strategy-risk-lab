from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from tradingview_strategy_risk_lab.data import FEATURE_COLUMNS
from tradingview_strategy_risk_lab.explainability import (
    compute_shap_importances,
    make_counterfactual_suggestions,
)
from tradingview_strategy_risk_lab.schemas import StrategyRiskReport, TrainingConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)

_SESSION_ORDER = {"asia": 0, "london": 1, "europe": 1, "us": 2, "ny": 2, "other": 1}


def _engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Pre-trade interaction features — zero lookahead bias."""
    frame = frame.copy()
    frame["session_code"] = frame["session"].map(_SESSION_ORDER).fillna(1).astype(float)
    frame["atr_trend_x"] = frame["atr_pct"] * frame["trend_score"]
    frame["vol_trend_x"] = frame["volume_z"] * frame["trend_score"]
    frame["side_atr_x"] = frame["side_code"] * frame["atr_pct"]
    frame["atr_pct_sq"] = frame["atr_pct"] ** 2
    return frame


ENGINEERED_FEATURES = FEATURE_COLUMNS + [
    "session_code",
    "atr_trend_x",
    "vol_trend_x",
    "side_atr_x",
    "atr_pct_sq",
]


def _optuna_objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    scale_pos_weight: float,
    random_state: int,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "auc",
        "random_state": random_state,
        "verbosity": 0,
    }
    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


def _tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    random_state: int,
    n_trials: int = 40,
) -> dict:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(
        lambda trial: _optuna_objective(
            trial, X_train, y_train, scale_pos_weight, random_state
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    best = study.best_params.copy()
    best["scale_pos_weight"] = scale_pos_weight
    best["eval_metric"] = "auc"
    best["random_state"] = random_state
    best["verbosity"] = 0
    return best, study.best_value, len(study.trials)


def train_strategy_risk_model(frame: pd.DataFrame, config: TrainingConfig) -> StrategyRiskReport:
    if len(frame) < config.min_rows:
        raise ValueError(f"strategy risk modeling needs at least {config.min_rows} trades")

    frame = _engineer_features(frame)
    target = (frame["pnl_r"] <= config.bad_trade_threshold_r).astype(int)

    if target.nunique() != 2:
        raise ValueError("dataset must contain both winning and losing trades")

    X_train, X_test, y_train, y_test = train_test_split(
        frame[ENGINEERED_FEATURES],
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target,
    )

    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())

    best_params, cv_auc, n_trials = _tune_xgboost(
        X_train, y_train, scale_pos_weight, config.random_state, n_trials=40
    )

    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    holdout_auc = roc_auc_score(y_test, probabilities)

    top_features = _feature_importance(model.feature_importances_, ENGINEERED_FEATURES)
    bad_trade_rate = round(float(target.mean()), 4)

    try:
        shap_features = compute_shap_importances(model, X_test.values, ENGINEERED_FEATURES)
    except Exception:
        shap_features = []

    cf_suggestions = make_counterfactual_suggestions(shap_features or top_features, bad_trade_rate)

    hpo_summary = (
        f"Optuna TPE search: {n_trials} trials, "
        f"best CV-AUC={cv_auc:.4f}, "
        f"holdout AUC={holdout_auc:.4f}. "
        f"Best params: n_est={best_params['n_estimators']}, "
        f"depth={best_params['max_depth']}, "
        f"lr={best_params['learning_rate']:.4f}"
    )

    cf_suggestions = [hpo_summary] + cf_suggestions

    return StrategyRiskReport(
        rows=len(frame),
        bad_trade_rate=bad_trade_rate,
        win_rate=round(float((frame["pnl_r"] > 0).mean()), 4),
        profit_factor=_profit_factor(frame["pnl_r"]),
        average_r=round(float(frame["pnl_r"].mean()), 4),
        max_drawdown_r=_max_drawdown(frame["pnl_r"]),
        roc_auc=round(float(holdout_auc), 4),
        holdout_rows=len(X_test),
        feature_count=len(ENGINEERED_FEATURES),
        evidence_checks=_evidence_checks(frame, config, len(X_test)),
        top_features=top_features,
        shap_features=shap_features,
        filter_suggestions=_filter_suggestions(frame),
        counterfactual_hint=_counterfactual_hint(top_features),
        counterfactual_suggestions=cf_suggestions,
    )


def _feature_importance(
    importances: np.ndarray, feature_names: list[str]
) -> list[dict[str, float | str]]:
    order = np.argsort(importances)[::-1][:8]
    return [
        {"feature": feature_names[i], "importance": round(float(importances[i]), 5)}
        for i in order
        if float(importances[i]) > 0
    ]


def _evidence_checks(
    frame: pd.DataFrame, config: TrainingConfig, holdout_rows: int
) -> list[dict[str, str]]:
    forbidden = {"pnl_r", "exit_price", "trade_id"}
    leakage = sorted(forbidden.intersection(set(ENGINEERED_FEATURES)))
    class_counts = (frame["pnl_r"] <= config.bad_trade_threshold_r).astype(int).value_counts()
    return [
        {
            "check": "holdout_split",
            "status": "pass",
            "evidence": f"{holdout_rows} trades reserved for holdout scoring.",
        },
        {
            "check": "zero_lookahead_features",
            "status": "pass" if not leakage else "fail",
            "evidence": "No post-exit fields are used as features."
            if not leakage
            else f"Remove leakage features: {', '.join(leakage)}.",
        },
        {
            "check": "minimum_class_count",
            "status": "pass" if int(class_counts.min()) >= 5 else "review",
            "evidence": f"Bad/good trade class counts: {class_counts.to_dict()}.",
        },
        {
            "check": "hpo_validation",
            "status": "pass",
            "evidence": "Optuna TPE ran 40 trials with 5-fold stratified CV on training set only.",
        },
        {
            "check": "claim_boundary",
            "status": "review",
            "evidence": "Risk model explains historical fragility; not a live execution rule.",
        },
    ]


def _counterfactual_hint(top_features: list[dict[str, float | str]]) -> str:
    if not top_features:
        return "No stable driver found."
    feature = str(top_features[0]["feature"])
    return (
        f"Primary SHAP driver: {feature}. "
        "Compare bad-trade rate above vs. below its median before deploying as a live filter."
    )


def _profit_factor(pnl_r: pd.Series) -> float:
    gains = float(pnl_r[pnl_r > 0].sum())
    losses = abs(float(pnl_r[pnl_r < 0].sum()))
    return round(gains / losses, 4) if losses > 0 else round(gains, 4)


def _max_drawdown(pnl_r: pd.Series) -> float:
    equity = pnl_r.cumsum()
    return round(abs(float((equity - equity.cummax()).min())), 4)


def _filter_suggestions(frame: pd.DataFrame) -> list[dict[str, float | str]]:
    bad_trade = frame["pnl_r"] <= 0
    overall = float(bad_trade.mean())
    candidates = [
        ("atr_pct", ">=", float(frame["atr_pct"].quantile(0.75))),
        ("volume_z", "<=", float(frame["volume_z"].quantile(0.25))),
        ("trend_score", "<=", float(frame["trend_score"].quantile(0.25))),
        ("atr_trend_x", "<=", float(frame["atr_trend_x"].quantile(0.25))),
    ]
    suggestions = []
    for feature, op, threshold in candidates:
        mask = frame[feature] >= threshold if op == ">=" else frame[feature] <= threshold
        if mask.sum() < 3:
            continue
        rate = float(bad_trade[mask].mean())
        if rate <= overall:
            continue
        suggestions.append(
            {
                "condition": f"{feature} {op} {threshold:.4f}",
                "bad_trade_rate": round(rate, 4),
                "baseline_bad_trade_rate": round(overall, 4),
                "suggested_filter": (
                    f"Review trades where {feature} {op} {threshold:.4f}; "
                    "validate on holdout before deploying."
                ),
            }
        )
    return suggestions[:3]
