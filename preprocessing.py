"""
preprocessing.py
-----------------
Implements Section 2.3 "Data Preprocessing" of the manuscript:
  1. Data cleaning (flag/report invalid entries -- automatic correction of
     entry errors was done by medical experts in the original study and is
     not something a script can safely replicate, so this step here focuses
     on detecting obviously invalid values, e.g. out-of-range binary flags,
     and reports them rather than silently guessing a fix).
  2. Missing value imputation using the mean of the corresponding feature.
  3. Min-Max scaling.
  4. Z-score standardization.

The manuscript applies Min-Max scaling *and then* Z-score standardization
(Section 2.3). Doing both in sequence is unusual (Z-scoring after Min-Max
scaling a feature to [0, 1] just re-centers/re-scales it again), but this
module follows the manuscript's stated order exactly for faithfulness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import BINARY_COLS, CONTINUOUS_COLS, FEATURE_COLS


def report_invalid_binary_values(df: pd.DataFrame) -> pd.DataFrame:
    """Flag binary columns containing values other than {0, 1, NaN}.

    Returns a small report DataFrame; does not modify `df`. Rows/values
    flagged here should be reviewed manually (as in the original study,
    where a medical expert adjudicated entry errors) rather than auto-fixed.
    """
    issues = []
    for col in BINARY_COLS:
        if col not in df.columns:
            continue
        bad_mask = ~df[col].isin([0, 1]) & df[col].notna()
        if bad_mask.any():
            issues.append(
                {"column": col, "n_invalid": int(bad_mask.sum()),
                 "invalid_values": df.loc[bad_mask, col].unique().tolist()}
            )
    return pd.DataFrame(issues)


def impute_missing_with_mean(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Impute missing values in `columns` using the column mean.

    Matches Section 2.3: "Missing values were identified and, when
    possible, imputed using the mean of the corresponding feature."
    """
    df = df.copy()
    for col in columns:
        if col in df.columns and df[col].isna().any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
    return df


def min_max_scale(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Apply Min-Max scaling to `columns`, returning the scaled df and the fitted scaler."""
    df = df.copy()
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


def z_score_standardize(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    """Apply Z-score standardization to `columns`, returning the scaled df and the fitted scaler."""
    df = df.copy()
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


def preprocess(
    raw_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    apply_minmax: bool = True,
    apply_zscore: bool = True,
) -> dict:
    """Run the full preprocessing pipeline described in Section 2.3.

    Parameters
    ----------
    raw_df : the raw patient-level DataFrame (must contain FEATURE_COLS + TARGET_COL)
    feature_cols : which columns to impute/scale (defaults to config.FEATURE_COLS)
    apply_minmax / apply_zscore : toggles, in case you only want one scaling step

    Returns
    -------
    dict with keys:
        "df"              -> fully preprocessed DataFrame
        "invalid_report"  -> report of out-of-range binary values found pre-cleaning
        "minmax_scaler"   -> fitted MinMaxScaler (or None)
        "zscore_scaler"   -> fitted StandardScaler (or None)
    """
    feature_cols = feature_cols or FEATURE_COLS
    invalid_report = report_invalid_binary_values(raw_df)

    df = impute_missing_with_mean(raw_df, feature_cols)

    minmax_scaler = None
    zscore_scaler = None

    if apply_minmax:
        df, minmax_scaler = min_max_scale(df, [c for c in feature_cols if c in df.columns])
    if apply_zscore:
        df, zscore_scaler = z_score_standardize(df, [c for c in feature_cols if c in df.columns])

    return {
        "df": df,
        "invalid_report": invalid_report,
        "minmax_scaler": minmax_scaler,
        "zscore_scaler": zscore_scaler,
    }
