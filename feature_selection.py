"""
feature_selection.py
---------------------
Implements Section 2.4.1 "Feature selection based on information functions"
"""

from __future__ import annotations
 
import pandas as pd
 
PRIMARY_PREDICTORS: list[str] = [
    "TG_Level_Before_Treatment",
    "TG_Level_Stage_1",
    "TG_Level_Stage_2",
]
 
 
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
    """Return every candidate feature ranked by mutual information, flagging
    which were selected as predictors."""
    ranked = mi_scores.sort_values(ascending=False)
    return pd.DataFrame(
        {
            "Feature": ranked.index,
            "Mutual_Information": ranked.values,
            "Selected_as_predictor": [f in set(selected) for f in ranked.index],
        }
    ).reset_index(drop=True)
