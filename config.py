"""
config.py
---------
Central configuration for the pipeline: column names, model hyperparameters,
and EWMA / GA settings. Values here are taken directly from the manuscript
(Tables 1, 2, 6 and 7) wherever the manuscript specifies them. Anywhere the
manuscript is silent (e.g. the exact DNN architecture, the k in k-fold CV),
a reasonable default is used and flagged with a comment -- check these
against your own original settings if you recall them, since the paper
text does not pin them down precisely.
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

# NOTE: the manuscript does not specify an architecture for the comparison
# "DNN" model beyond calling it a deep neural network distinct from the MLP.
# It is implemented here as a deeper MLPClassifier (3 hidden layers) so that
# it is architecturally distinguishable from the single-hidden-layer MLP.
# Swap this out for your original architecture if you recall it.
DNN_PARAMS = dict(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
)

# NOTE: k for k-fold cross-validation is not stated in the manuscript text.
# 5-fold is the most common default and is used here.
CV_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# EWMA control chart defaults (manuscript Table 7 / Eqs. 8-10)
# ---------------------------------------------------------------------------
EWMA_LAMBDA_INITIAL = 0.15
EWMA_L_INITIAL = 3.5

# ---------------------------------------------------------------------------
# Genetic algorithm settings (manuscript Section 2.6)
# The manuscript describes the mechanism (population, fitness = number of
# out-of-control points, selection/crossover/mutation, generations) but does
# not give exact population size / generation count / mutation rate. Sensible
# defaults are used below -- tune as needed.
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
