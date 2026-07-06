"""
main.py
-------
End-to-end pipeline reproducing the methodology in:

  "A genetic algorithm-optimized EWMA control chart integrating machine
  learning-based risk prediction for multistage thyroid cancer treatment
  monitoring"

Usage
-----
    python main.py --data path/to/your_dataset.csv --outdir results/

Your CSV must contain the columns listed in config.FEATURE_COLS plus the
target column config.TARGET_COL ("Outcome"), matching Table 2 of the
manuscript. If you no longer have your original data file either, see
`make_synthetic_dataset.py` in this folder to generate a placeholder
dataset with the same schema so you can sanity-check that the pipeline runs
end to end -- do NOT use synthetic data to produce numbers for the paper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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

    # ---- 4. Predicted risk scores + EWMA baseline stats (Section 2.5) -----
    risk_scores = predict_risk_scores(best_model, X)

    chart_stats = []
    charts_initial = {}
    charts_optimized = {}
    ga_results = {}

    for feature in CONTROL_CHART_FEATURES:
        if feature not in df.columns:
            continue
        series = df[feature].to_numpy()  # preprocessed (scaled) feature series
        mu, sigma = series.mean(), series.std(ddof=0)
        chart_stats.append({"Feature": feature, "Mean (mu)": mu, "Standard Deviation (sigma)": sigma})

        # Initial (non-optimized) chart -- Table 6 / Figure 4 orange line
        chart_initial = build_ewma_chart(series, mu, sigma, EWMA_LAMBDA_INITIAL, EWMA_L_INITIAL)
        charts_initial[feature] = chart_initial

        # ---- 5. GA optimization of (lambda, L) (Section 2.6, Table 7) -----
        ga_result = optimize_ewma_parameters(series, mu, sigma)
        ga_results[feature] = ga_result
        chart_optimized = build_ewma_chart(
            series, mu, sigma, ga_result.best_lambda, ga_result.best_L
        )
        charts_optimized[feature] = chart_optimized

    pd.DataFrame(chart_stats).to_csv(outdir / "table5_risk_score_stats.csv", index=False)

    ga_summary = pd.DataFrame(
        {
            feature: {
                "Initial lambda": EWMA_LAMBDA_INITIAL,
                "Initial L": EWMA_L_INITIAL,
                "Optimized lambda": res.best_lambda,
                "Optimized L": res.best_L,
                "Out-of-control count (optimized)": res.best_fitness,
            }
            for feature, res in ga_results.items()
        }
    ).T
    ga_summary.to_csv(outdir / "table7_ga_optimized_parameters.csv")
    print("\nTable 7 -- GA-optimized parameters per feature:")
    print(ga_summary)

    # ---- 6. Figure 4 -- before/after comparison plots ----------------------
    fig, axes = plt.subplots(len(charts_initial), 1, figsize=(8, 4 * len(charts_initial)))
    if len(charts_initial) == 1:
        axes = [axes]

    for ax, feature in zip(axes, charts_initial.keys()):
        c_init = charts_initial[feature]
        c_opt = charts_optimized[feature]
        ax.plot(c_init["ewma"], color="orange", label="Initial (orange)")
        ax.plot(c_opt["ewma"], color="green", label="GA-optimized (green)")
        ax.plot(c_init["ucl"], color="red", linestyle="--", linewidth=0.8)
        ax.plot(c_init["lcl"], color="red", linestyle="--", linewidth=0.8)
        ax.set_title(f"EWMA Control Chart -- {feature}")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
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
