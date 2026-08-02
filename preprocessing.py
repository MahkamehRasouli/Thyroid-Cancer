"""Section 2.3: data cleaning check, mean imputation, Min-Max scaling, Z-score standardization."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from config import BINARY_COLS, FEATURE_COLS


def report_invalid_binary_values(df: pd.DataFrame) -> pd.DataFrame:
    """Flag binary columns containing values other than {0, 1, NaN}; does not modify df.
    Flagged values should be reviewed manually rather than auto-corrected."""
    issues = []
    for col in BINARY_COLS:
        if col not in df.columns:
            continue
        bad_mask = ~df[col].isin([0, 1]) & df[col].notna()
        if bad_mask.any():
            issues.append(
                {
                    "column": col,
                    "n_invalid": int(bad_mask.sum()),
                    "invalid_values": df.loc[bad_mask, col].unique().tolist(),
                }
            )
    return pd.DataFrame(issues)


def impute_missing_with_mean(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Impute missing values in `columns` with the column mean."""
    df = df.copy()
    for col in columns:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    return df


def min_max_scale(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Min-Max scale `columns`; return the scaled df and fitted scaler."""
    df = df.copy()
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


def z_score_standardize(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    """Z-score standardize `columns`; return the scaled df and fitted scaler."""
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
    """Run the Section 2.3 pipeline: invalid-value report, mean imputation, and scaling.

    Returns a dict with keys: df, invalid_report, minmax_scaler, zscore_scaler.
    """
    feature_cols = feature_cols or FEATURE_COLS
    invalid_report = report_invalid_binary_values(raw_df)
    df = impute_missing_with_mean(raw_df, feature_cols)

    minmax_scaler = zscore_scaler = None
    present = [c for c in feature_cols if c in df.columns]
    if apply_minmax:
        df, minmax_scaler = min_max_scale(df, present)
    if apply_zscore:
        df, zscore_scaler = z_score_standardize(df, present)

    return {
        "df": df,
        "invalid_report": invalid_report,
        "minmax_scaler": minmax_scaler,
        "zscore_scaler": zscore_scaler,
    }
