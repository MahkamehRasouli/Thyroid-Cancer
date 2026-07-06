# GA-Optimized EWMA Control Chart with MLP Risk Prediction — Reconstructed Code

This is a reconstruction of the analysis pipeline described in:

> "A genetic algorithm–optimized EWMA control chart integrating machine
> learning–based risk prediction for multistage thyroid cancer treatment
> monitoring" (PLOS ONE, manuscript PONE-D-26-16204).

It was rebuilt from the manuscript's Methods section (data preprocessing,
mutual-information feature selection, MLP risk prediction, EWMA control
charts, and GA parameter optimization), since the original code was lost.
**Read the "Assumptions and known gaps" section below before treating this
as a drop-in replacement for the original implementation** — anywhere the
manuscript didn't fully specify a detail, a reasonable default was chosen
and flagged, and you should overwrite those with your actual original
settings if you recall them, or re-tune them against your own results.

## Files

| File | Implements |
|---|---|
| `config.py` | Column schema (Table 2), MLP hyperparameters (Table 1), GA/EWMA settings |
| `preprocessing.py` | Section 2.3: cleaning check, mean imputation, Min-Max scaling, Z-score standardization |
| `feature_selection.py` | Section 2.4.1: mutual information (Eqs. 1–2), Table 3 |
| `models.py` | Section 2.4: MLP + 5 comparison models, Table 4 metrics |
| `ewma.py` | Section 2.5: EWMA statistic (Eq. 8), control limits (Eqs. 9–10) |
| `genetic_algorithm.py` | Section 2.6: GA optimization of λ and L |
| `main.py` | Orchestrates the full pipeline, writes Tables 3/4/5/7 and a Figure-4-style plot |
| `make_synthetic_dataset.py` | Generates a placeholder dataset (same schema as Table 2) for smoke-testing only |

## Usage

```bash
pip install -r requirements.txt

# Run on your real, de-identified dataset (must have the columns in config.py / Table 2):
python main.py --data your_dataset.csv --outdir results/

# Or smoke-test the pipeline with placeholder data first:
python make_synthetic_dataset.py --out synthetic_data.csv --n 80
python main.py --data synthetic_data.csv --outdir results/
```

Outputs land in `results/`: `table3_mutual_information.csv`,
`table4_model_comparison.csv`, `table5_risk_score_stats.csv`,
`table7_ga_optimized_parameters.csv`, and `figure4_ewma_comparison.png`.

This was smoke-tested end-to-end on synthetic data and runs cleanly; the
synthetic run's numbers are meaningless (random data) and only confirm the
code executes without errors.

## Assumptions and known gaps (check these against your original work)

The manuscript specifies some things precisely and leaves others
unspecified. Exact numeric results in your Tables 3–7 depend on your real
data and on the exact settings you originally used, which this
reconstruction cannot recover on its own:

1. **k-fold CV**: the manuscript reports "k-fold cross-validation accuracy"
   without stating k. This code defaults to `CV_FOLDS = 5` in `config.py`.
2. **DNN architecture**: Table 1 pins down the MLP's hyperparameters
   exactly, but the manuscript never specifies the "DNN" comparison
   model's architecture beyond calling it a deep neural network. This code
   uses a 3-hidden-layer `MLPClassifier((64, 32, 16))` to differentiate it
   from the single-hidden-layer MLP — substitute your original
   architecture if you recall it (e.g., if you used TensorFlow/Keras
   instead of scikit-learn).
3. **Feature-selection cutoff**: Table 3 lists mutual information scores
   but the manuscript doesn't state a numeric threshold or fixed top-k used
   to decide "primary predictors" — it says the three thyroglobulin
   features had the highest scores and were used. `feature_selection.py`
   exposes both a `top_k` and a `threshold` option; pick whichever matches
   what you actually did.
4. **GA settings**: population size, number of generations, mutation rate,
   and crossover mechanism are described qualitatively in Section 2.6
   (tournament/roulette selection, crossover, mutation) but not
   numerically. Defaults are in `config.py` — tune these if your optimized
   λ/L values (Table 7: λ=0.20, L=4.0) don't come out close to what you
   originally reported.
5. **Fitness function caveat**: the manuscript defines GA fitness purely as
   "number of out-of-control observations," with lower being better. In
   isolation, that objective can trivially reward pushing `L` toward its
   upper bound (wider limits → fewer flagged points) without any offsetting
   pressure to preserve sensitivity. If you see this in your own runs (GA
   converging to a very large `L` with zero out-of-control points for every
   feature), you likely had — or should add — a secondary term or
   constraint that penalizes excessively wide limits, even if the
   manuscript text doesn't spell one out explicitly.
6. **Order of Min-Max scaling then Z-score standardization** (Section 2.3):
   implemented exactly in that sequence per the text, even though applying
   Z-score after Min-Max is somewhat redundant statistically — flagging in
   case your original pipeline applied only one of the two.
7. **Preprocessing "data cleaning"**: the manuscript describes manual
   review/correction by medical experts for entry errors. This isn't
   something a script can safely automate, so `preprocessing.py` only
   *flags* out-of-range binary values for your manual review rather than
   auto-correcting them.

## Data note

This code expects a CSV with the column names in `config.py`
(`FEATURE_COLS` + `Outcome`), matching Table 2. It does not include or
require your original patient data — you'll need to supply that yourself
when running it for real results.
