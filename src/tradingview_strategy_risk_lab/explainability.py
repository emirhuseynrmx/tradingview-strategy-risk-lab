from __future__ import annotations

import pandas as pd


def shap_and_dice_stack_note() -> dict[str, str]:
    """Document the intended explainability boundary without running heavy explainers in tests."""
    return {
        "shap": "Use shap.TreeExplainer on the trained RandomForestClassifier for global drivers.",
        "dice": "Use dice_ml.Data and dice_ml.Model to generate counterfactual trade conditions.",
    }


def make_driver_table(top_features: list[dict[str, float | str]]) -> pd.DataFrame:
    return pd.DataFrame(top_features)

