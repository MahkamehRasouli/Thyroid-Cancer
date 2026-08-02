"""Section 2.4.1: mutual-information feature scoring (Eqs. 1-2) and predictor selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRIMARY_PREDICTORS: list[str] = [
    "TG_Level_Before_Treatment",
    "TG_Level_Stage_1",
    "TG_Level_Stage_2",
]


def _mutual_information(x: pd.Series, y: pd.Series) -> float:
    """Discrete mutual information I(X;Y) in bits (Eq. 2) from empirical frequencies."""
    joint = pd.crosstab(x, y)
    p_xy = np.array(joint, dtype=float)
    p_xy = p_xy / p_xy.sum()
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = p_xy * np.log2(p_xy / (p_x * p_y))
    return float(np.nansum(terms))


def compute_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    continuous_cols: list[str] | None = None,
    n_bins: int = 5,
) -> pd.Series:
    """Mutual information between each feature and the outcome, returned as a Series
    sorted in descending order (Table 3).

    Continuous features (given in `continuous_cols`) are discretized into `n_bins`
    equal-frequency bins before computing discrete mutual information; binary and
    categorical features are used as-is. Values are in bits.
    """
    continuous_cols = continuous_cols or []
    scores: dict[str, float] = {}
    for col in X.columns:
        feature = X[col]
        if col in continuous_cols:
            feature = pd.qcut(feature, q=n_bins, duplicates="drop")
        scores[col] = _mutual_information(feature, y)
    return pd.Series(scores).sort_values(ascending=False)


def select_features(
    mi_scores: pd.Series,
    method: str = "explicit",
    features: list[str] | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[str]:
    """Select predictor features.

    method="explicit"  -> return `features` (defaults to PRIMARY_PREDICTORS).
    method="top_k"      -> return the `top_k` features by descending MI.
    method="threshold"  -> return features with MI >= `threshold`.
    """
    if method == "explicit":
        chosen = features if features is not None else PRIMARY_PREDICTORS
        if not chosen:
            raise ValueError("method='explicit' requires a non-empty `features` list.")
        missing = [f for f in chosen if f not in mi_scores.index]
        if missing:
            raise ValueError(f"Selected features not present in mi_scores: {missing}")
        return list(chosen)

    if method == "top_k":
        if top_k is None:
            raise ValueError("method='top_k' requires `top_k`.")
        return mi_scores.sort_values(ascending=False).head(top_k).index.tolist()

    if method == "threshold":
        if threshold is None:
            raise ValueError("method='threshold' requires `threshold`.")
        return mi_scores[mi_scores >= threshold].index.tolist()

    raise ValueError(f"Unknown method: {method!r}")


def selection_report(mi_scores: pd.Series, selected: list[str]) -> pd.DataFrame:
    """Every candidate feature ranked by mutual information, flagging which were selected."""
    ranked = mi_scores.sort_values(ascending=False)
    return pd.DataFrame(
        {
            "Feature": ranked.index,
            "Mutual_Information": ranked.values,
            "Selected_as_predictor": [f in set(selected) for f in ranked.index],
        }
    ).reset_index(drop=True)
