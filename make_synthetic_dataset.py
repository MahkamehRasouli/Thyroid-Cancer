"""
make_synthetic_dataset.py
--------------------------
Generates a test dataset with the same columns, ranges, and rough
summary statistics as Table 2 of the manuscript (N=80 patients). This is
for testing the pipeline only and it does not reproduce the paper's
actual results, since it is not the paper's actual data.

Usage:
    python make_synthetic_dataset.py --out synthetic_data.csv --n 80
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def make_synthetic_dataset(n: int = 80, random_state: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    df = pd.DataFrame(
        {
            "Age": rng.normal(50.74, 18.06, n).clip(21, 79),
            "Gender": rng.binomial(1, 0.54, n),
            "BMI": rng.normal(26.58, 4.71, n).clip(18.6, 34.8),
            "Smoking_Status": rng.binomial(1, 0.54, n),
            "Alcohol_Consumption": rng.binomial(1, 0.54, n),
            "Comorbidities": rng.poisson(1.93, n).clip(0, 4),
            "Family_History": rng.binomial(1, 0.40, n),
            "Previous_Thyroid_Operation": rng.binomial(1, 0.46, n),
            "Tumor_Size": rng.normal(2.92, 1.17, n).clip(1.1, 5.0),
            "Surgery_Type": rng.binomial(1, 0.56, n),
            "TG_Level_Before_Treatment": rng.normal(5.2, 2.97, n).clip(0.2, 10.0),
            "TG_Level_Stage_1": rng.normal(4.16, 2.64, n).clip(0.3, 9.9),
            "TG_Level_Stage2": rng.normal(5.28, 2.8, n).clip(0.1, 9.8),
            "Time_Between_Stages": rng.normal(104.48, 43.45, n).clip(30, 179),
            "Radiation_Therapy": rng.binomial(1, 0.46, n),
            "Radioactive_Iodine_Dose": rng.normal(65.8, 19.64, n).clip(33.6, 99.3),
        }
    )

    # Synthetic outcome correlated with TG levels, purely for smoke-testing
    # the classifiers -- has no relationship to the real study's findings.
    risk_signal = (
        0.4 * df["TG_Level_Stage2"]
        - 0.3 * df["TG_Level_Before_Treatment"]
        + rng.normal(0, 2, n)
    )
    df["Outcome"] = (risk_signal > np.median(risk_signal)).astype(int)

    return df


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate a synthetic placeholder dataset for smoke-testing.")
    parser.add_argument("--out", default="synthetic_data.csv")
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = make_synthetic_dataset(args.n, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote synthetic dataset with {len(df)} rows to {args.out}")
