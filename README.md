# GA-Optimized EWMA Control Chart with MLP Risk Prediction

This is the analysis pipeline described in:

> "A genetic algorithm–optimized EWMA control chart integrating machine
> learning–based risk prediction for multistage thyroid cancer treatment
> monitoring" (PLOS ONE, manuscript PONE-D-26-16204).

## Files

| File | Implements |
|---|---|
| `config.py` | Column schema (Table 2), MLP hyperparameters (Table 1), GA/EWMA settings |
| `preprocessing.py` | Section 2.3: cleaning check, mean imputation, Min-Max scaling, Z-score standardization |
| `feature_selection.py` | Section 2.4.1: mutual information (Eqs. 1–2), Table 3 |
| `models.py` | Section 2.4: MLP + 5 comparison models, Table 4 metrics |
| `ewma.py` | Section 2.5: EWMA statistic (Eq. 8), control limits (Eqs. 9–10) |
| `genetic_algorithm.py` | Section 2.6: GA optimization of λ and L |
| `main.py` | Orchestrates the full pipeline, writes Tables 3/4/5/7 and Figure 4 plot |
| `make_synthetic_dataset.py` | Generates a placeholder dataset (same schema as Table 2) for testing only |

## Usage

```bash
pip install -r requirements.txt

python main.py --data your_dataset.csv --outdir results/

# test the pipeline with placeholder data:
python make_synthetic_dataset.py --out synthetic_data.csv --n 80
python main.py --data synthetic_data.csv --outdir results/
```

Outputs land in `results/`: `table3_mutual_information.csv`,
`table4_model_comparison.csv`, `table5_risk_score_stats.csv`,
`table7_ga_optimized_parameters.csv`, and `figure4_ewma_comparison.png`.
