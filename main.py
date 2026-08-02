"""
main.py
-------
End-to-end pipeline for:
"A genetic algorithm-optimized EWMA control chart integrating machine
 learning-based risk prediction for multistage thyroid cancer treatment
 monitoring"
 

Usage
-----
    python main.py --data path/to/your_dataset.csv --outdir results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CONTROL_CHART_FEATURES,
    EWMA_L_INITIAL,
    EWMA_LAMBDA_INITIAL,
    FEATURE_COLS,
    TARGET_COL,
)
from ewma import build_ewma_chart
from feature_selection import compute_mutual_information
from genetic_algorithm import optimize_ewma_parameters
from models import compare_models, predict_risk_scores
from preprocessing import preprocess


RISK_SCORE_MODE = "per_feature"


def _chart_one_series(series: np.ndarray, name: str):
    series = np.asarray(series, dtype=float)
    mu, sigma = series.mean(), series.std(ddof=0)

    chart_initial = build_ewma_chart(series, mu, sigma, EWMA_LAMBDA_INITIAL, EWMA_L_INITIAL)
    ga_result = optimize_ewma_parameters(series, mu, sigma)
    chart_optimized = build_ewma_chart(series, mu, sigma, ga_result.best_lambda, ga_result.best_L)

    stats = {"Series": name, "Mean (mu)": mu, "Standard Deviation (sigma)": sigma}
    return stats, chart_initial, chart_optimized, ga_result


def run_pipeline(data_path: str, outdir: str) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load + preprocess (Section 2.2, 2.3) --------------------------
    raw_df = pd.read_csv(data_path)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing expected columns: {missing}")

    prep = preprocess(raw_df)
    df = prep["df"]
    if not prep["invalid_report"].empty:
        print("Invalid binary values detected (review before proceeding):")
        print(prep["invalid_report"])

    X = df[FEATURE_COLS]
    y = raw_df[TARGET_COL]  # target is a class label, left un-scaled

    # ---- 2. Feature selection via mutual information (Section 2.4.1) ------
    mi_scores = compute_mutual_information(X, y)
    mi_scores.to_csv(outdir / "table3_mutual_information.csv")
    print("\nTable 3 -- Mutual information scores:")
    print(mi_scores)

    # ---- 3. Risk prediction modeling (Section 2.4, Table 4) ---------------
    results_df, fitted_models = compare_models(X, y)
    results_df.to_csv(outdir / "table4_model_comparison.csv")
    print("\nTable 4 -- Model comparison:")
    print(results_df)

    best_model_name = results_df["AUC"].astype(float).idxmax()
    best_model = fitted_models[best_model_name]
    print(f"\nBest model by AUC: {best_model_name}")

    # ---- 4. Predicted risk scores -> monitored series (Section 2.5) -------
    # CORRECTED: the EWMA monitors the MLP-derived risk values, not raw features.
    risk_scores = predict_risk_scores(best_model, X)
    risk_scores = np.asarray(risk_scores, dtype=float)

    chart_stats = []
    charts_initial = {}
    charts_optimized = {}
    ga_results = {}

    if RISK_SCORE_MODE == "per_patient":
        if risk_scores.ndim != 1:
            raise ValueError(
                "RISK_SCORE_MODE='per_patient' expects a 1-D risk-score array "
                f"(one value per patient); got shape {risk_scores.shape}."
            )
        stats, c_init, c_opt, ga = _chart_one_series(risk_scores, "MLP risk score")
        chart_stats.append(stats)
        charts_initial["MLP risk score"] = c_init
        charts_optimized["MLP risk score"] = c_opt
        ga_results["MLP risk score"] = ga

    elif RISK_SCORE_MODE == "per_feature":
        risk_df = pd.DataFrame(risk_scores, columns=CONTROL_CHART_FEATURES) \
            if risk_scores.ndim == 2 and risk_scores.shape[1] == len(CONTROL_CHART_FEATURES) \
            else None
        if risk_df is None:
            raise ValueError(
                "RISK_SCORE_MODE='per_feature' expects risk_scores with shape "
                f"(n_patients, {len(CONTROL_CHART_FEATURES)}) aligned to "
                f"CONTROL_CHART_FEATURES; got shape {risk_scores.shape}. "
                "Update predict_risk_scores() to return per-feature MLP risk values."
            )
        for feature in CONTROL_CHART_FEATURES:
            stats, c_init, c_opt, ga = _chart_one_series(risk_df[feature].to_numpy(), feature)
            chart_stats.append(stats)
            charts_initial[feature] = c_init
            charts_optimized[feature] = c_opt
            ga_results[feature] = ga
    else:
        raise ValueError(f"Unknown RISK_SCORE_MODE: {RISK_SCORE_MODE!r}")

    # ---- 5. Tables 5 and 7 ------------------------------------------------
    pd.DataFrame(chart_stats).to_csv(outdir / "table5_risk_score_stats.csv", index=False)

    ga_summary = pd.DataFrame(
        {
            name: {
                "Initial lambda": EWMA_LAMBDA_INITIAL,
                "Initial L": EWMA_L_INITIAL,
                "Optimized lambda": res.best_lambda,
                "Optimized L": res.best_L,
                "Fitness (optimized)": res.best_fitness,
            }
            for name, res in ga_results.items()
        }
    ).T
    ga_summary.to_csv(outdir / "table7_ga_optimized_parameters.csv")
    print("\nTable 7 -- GA-optimized parameters:")
    print(ga_summary)

    # ---- 6. Figure 4 -- before/after comparison plots ---------------------
    n = len(charts_initial)
    fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, charts_initial.keys()):
        c_init = charts_initial[name]
        c_opt = charts_optimized[name]
        ax.plot(c_init["ewma"], color="orange", label=f"Original ($\\lambda$={EWMA_LAMBDA_INITIAL}, L={EWMA_L_INITIAL})")
        ax.plot(c_opt["ewma"], color="green", label="GA-optimized")
        ax.plot(c_init["ucl"], color="red", linestyle="--", linewidth=0.8, label="Control limits")
        if "lcl" in c_init:
            ax.plot(c_init["lcl"], color="red", linestyle="--", linewidth=0.8)
        ax.set_title(f"EWMA Control Chart -- {name} (MLP risk values)")
        ax.set_xlabel("Patient index")
        ax.set_ylabel("MLP risk value")
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "figure4_ewma_comparison.png", dpi=150)
    print(f"\nSaved comparison figure to {outdir / 'figure4_ewma_comparison.png'}")
    print(f"\nAll outputs written to: {outdir.resolve()}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the GA-optimized EWMA + MLP risk pipeline.")
    parser.add_argument("--data", required=True, help="Path to input CSV (see config.py for required columns).")
    parser.add_argument("--outdir", default="results", help="Directory to write output tables/figures to.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(args.data, args.outdir)
