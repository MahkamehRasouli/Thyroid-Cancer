# GA-Optimized EWMA Control Chart with MLP Risk Prediction

Analysis pipeline for the study:

> "A genetic algorithm–optimized EWMA control chart integrating machine
> learning–based risk prediction for multistage thyroid cancer treatment
> monitoring" (PLOS ONE, manuscript PONE-D-26-16204).

The pipeline performs mutual-information feature selection, trains an MLP
risk-prediction model, monitors the model-derived risk values with EWMA
control charts, and optimizes the EWMA parameters (λ, L) with a genetic
algorithm.

## Data availability

The clinical data underlying this study cannot be shared publicly, as the
data use agreement with the data-owning institution prohibits transfer or
public deposition of the patient records. To allow the pipeline to be run
and inspected without the original data, `make_synthetic_dataset.py`
generates a synthetic dataset with the same schema (Table 2). Results
produced from synthetic data are for functional testing only and do not
reproduce the values reported in the manuscript.

## Files

| File | Implements |
|---|---|
| `config.py` | Column schema (Table 2), MLP hyperparameters (Table 1), GA/EWMA settings |
| `preprocessing.py` | Section 2.3: cleaning checks, mean imputation, Min-Max scaling, Z-score standardization |
| `feature_selection.py` | Section 2.4.1: mutual information (Eqs. 1–2), Table 3; predictor selection combining MI ranking with clinical input |
| `models.py` | Section 2.4: MLP and five comparison models, Table 4 metrics |
| `ewma.py` | Section 2.5: EWMA statistic (Eq. 8), control limits (Eqs. 9–10) |
| `genetic_algorithm.py` | Section 2.6: GA optimization of λ and L |
| `main.py` | Orchestrates the full pipeline; writes Tables 3/4/5/7 and the Figure 4 plot |
| `make_synthetic_dataset.py` | Generates a synthetic dataset (same schema as Table 2) for testing |

## Usage

```bash
pip install -r requirements.txt

# run on your data
python main.py --data your_dataset.csv --outdir results/

# or test the pipeline with synthetic data
python make_synthetic_dataset.py --out synthetic_data.csv --n 80
python main.py --data synthetic_data.csv --outdir results/
```

## Outputs

Written to `results/`:
`table3_mutual_information.csv`, `table4_model_comparison.csv`,
`table5_risk_score_stats.csv`, `table7_ga_optimized_parameters.csv`,
and `figure4_ewma_comparison.png`.
python main.py --data your_dataset.csv --outdir results/

# test the pipeline with placeholder data:
python make_synthetic_dataset.py --out synthetic_data.csv --n 80
python main.py --data synthetic_data.csv --outdir results/
```

Outputs land in `results/`: `table3_mutual_information.csv`,
`table4_model_comparison.csv`, `table5_risk_score_stats.csv`,
`table7_ga_optimized_parameters.csv`, and `figure4_ewma_comparison.png`.
