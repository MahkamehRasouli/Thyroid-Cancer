"""
config.py
---------
Central configuration for the pipeline: column names, model hyperparameters,
and EWMA / GA settings.
"""

# ---------------------------------------------------------------------------
# Dataset schema (Table 2 of the manuscript)
# ---------------------------------------------------------------------------
TARGET_COL = "Outcome"  # 0 = Unsuccessful, 1 = Successful

# Binary / categorical (already 0/1 coded in the source data per Table 2)
BINARY_COLS = [
    "Gender",                      # 0 = Female, 1 = Male
    "Smoking_Status",               # 0 = No, 1 = Yes
    "Alcohol_Consumption",          # 0 = No, 1 = Yes
    "Family_History",               # 0 = No, 1 = Yes
    "Previous_Thyroid_Operation",   # 0 = No, 1 = Yes
    "Surgery_Type",                 # 0 = Partial, 1 = Total
    "Radiation_Therapy",            # 0 = No, 1 = Yes
]

# Continuous / count variables
CONTINUOUS_COLS = [
    "Age",
    "BMI",
    "Comorbidities",
    "Tumor_Size",
    "TG_Level_Before_Treatment",
    "TG_Level_Stage_1",
    "TG_Level_Stage2",
    "Time_Between_Stages",
    "Radioactive_Iodine_Dose",
]

FEATURE_COLS = BINARY_COLS + CONTINUOUS_COLS

# Features used to build the EWMA control charts (Tables 5 and 6)
CONTROL_CHART_FEATURES = [
    "TG_Level_Before_Treatment",
    "TG_Level_Stage_1",
    "TG_Level_Stage2",
    "Tumor_Size",
    "Time_Between_Stages",
    "Radioactive_Iodine_Dose",
]

# ---------------------------------------------------------------------------
# MLP hyperparameters (manuscript Table 1, verbatim)
# ---------------------------------------------------------------------------
MLP_PARAMS = dict(
    hidden_layer_sizes=(100,),   # "One hidden layer with 100 neurons"
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=500,                # "Number of Epochs" -> sklearn max_iter
    random_state=42,
)

DNN_PARAMS = dict(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
)

CV_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# EWMA control chart (manuscript Table 7 / Eqs. 8-10)
# ---------------------------------------------------------------------------
EWMA_LAMBDA_INITIAL = 0.15
EWMA_L_INITIAL = 3.5

# ---------------------------------------------------------------------------
# Genetic algorithm settings (manuscript Section 2.6)
# ---------------------------------------------------------------------------
GA_LAMBDA_BOUNDS = (0.01, 1.0)
GA_L_BOUNDS = (1.0, 5.0)
GA_POPULATION_SIZE = 40
GA_GENERATIONS = 100
GA_TOURNAMENT_SIZE = 3
GA_CROSSOVER_RATE = 0.8
GA_MUTATION_RATE = 0.2
GA_MUTATION_STD = 0.1
GA_ELITISM = 2
GA_RANDOM_STATE = 42
