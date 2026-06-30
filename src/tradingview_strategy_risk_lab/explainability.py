from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def shap_and_dice_stack_note() -> dict[str, str]:
    return {
        "shap": "TreeExplainer runs on the trained RandomForestClassifier for global drivers.",
        "dice": "Use dice_ml.Data and dice_ml.Model to generate counterfactual trade conditions.",
    }


def compute_shap_importances(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, float | str]]:
    import shap

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)
    sv = np.asarray(sv)
    # Newer SHAP returns (n_samples, n_features, n_classes); older returns list[array]
    if sv.ndim == 3:
        values = sv[:, :, 1]  # positive class
    elif sv.ndim == 2:
        values = sv  # regression or single-output
    else:
        # List format [class0_array, class1_array]
        values = np.asarray(sv[1])

    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return [
        {
            "feature": feature_names[i],
            "shap_importance": round(float(mean_abs[i]), 6),
        }
        for i in order
        if float(mean_abs[i]) > 0
    ]


def make_counterfactual_suggestions(
    top_shap: list[dict[str, float | str]],
    bad_trade_rate: float,
) -> list[str]:
    if not top_shap:
        return ["Collect more trade history before generating counterfactual suggestions."]

    driver = str(top_shap[0]["feature"])
    suggestions = [
        f"Primary driver is '{driver}' — compare bad-trade rate above vs. below its median.",
        f"Overall bad-trade rate is {bad_trade_rate:.1%}. Tighten '{driver}' filter and measure "
        "holdout impact before deploying.",
    ]
    if len(top_shap) >= 2:
        second = str(top_shap[1]["feature"])
        suggestions.append(
            f"Secondary driver '{second}' is orthogonal — test a compound filter "
            f"({driver} AND {second}) on out-of-sample data."
        )
    suggestions.append(
        "Keep immutable trade context (symbol, session) fixed; only adjust actionable "
        "setup conditions when testing counterfactuals."
    )
    return suggestions


def make_driver_table(top_features: list[dict[str, float | str]]) -> pd.DataFrame:
    return pd.DataFrame(top_features)


def format_driver_block(top_features: list[dict[str, float | str]], use_shap: bool = False) -> str:
    key = "shap_importance" if use_shap else "importance"
    label = "SHAP importance" if use_shap else "RF importance"
    lines = [f"  {'Feature':<22} {label}"]
    lines.append("  " + "-" * 38)
    for item in top_features[:5]:
        feat = str(item["feature"])
        val = float(item.get(key, item.get("importance", 0)))
        lines.append(f"  {feat:<22} {val:.5f}")
    return "\n".join(lines)
