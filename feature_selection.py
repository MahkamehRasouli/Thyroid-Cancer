"""
feature_selection.py
---------------------
Implements Section 2.4.1 "Feature selection based on information functions":
mutual information between each candidate feature and the target outcome
(Eqs. 1-2). sklearn's mutual_info_classif implements the same entropy /
mutual-information concept described in the manuscript for a categorical
target.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from config import RANDOM_STATE


def compute_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    discrete_features: list[bool] | str = "auto",
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """Compute mutual information between each column of X and y.

    Returns a Series indexed by feature name, sorted descending -- directly
    analogous to Table 3 in the manuscript.
    """
    mi = mutual_info_classif(
        X, y, discrete_features=discrete_features, random_state=random_state
    )
    return pd.Series(mi, index=X.columns, name="Mutual_Information").sort_values(
        ascending=False
    )


def select_top_features(
    mi_scores: pd.Series,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[str]:
    """Select features either by top-k or by a minimum mutual-information threshold.

    Provide exactly one of `top_k` or `threshold`. The manuscript selected
    thyroglobulin-related features as "primary predictors" based on having
    the highest mutual information (Table 3) -- i.e. a top-k-style choice --
    but did not report a fixed k or numeric cutoff, so both options are
    exposed here for you to match your original choice.
    """
    if (top_k is None) == (threshold is None):
        raise ValueError("Provide exactly one of `top_k` or `threshold`.")
    if top_k is not None:
        return mi_scores.head(top_k).index.tolist()
    return mi_scores[mi_scores >= threshold].index.tolist()
