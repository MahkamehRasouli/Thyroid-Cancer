"""Central configuration: column schema, model hyperparameters, and EWMA/GA settings."""

TARGET_COL = "Outcome"

BINARY_COLS = [
    "Gender",
    "Smoking_Status",
    "Alcohol_Consumption",
    "Family_History",
    "Previous_Thyroid_Operation",
    "Surgery_Type",
    "Radiation_Therapy",
]

CONTINUOUS_COLS = [
    "Age",
    "BMI",
    "Comorbidities",
    "Tumor_Size",
    "TG_Level_Before_Treatment",
    "TG_Level_Stage_1",
    "TG_Level_Stage_2",
    "Time_Between_Stages",
    "Radioactive_Iodine_Dose",
]

FEATURE_COLS = BINARY_COLS + CONTINUOUS_COLS

CONTROL_CHART_FEATURES = [
    "TG_Level_Before_Treatment",
    "TG_Level_Stage_1",
    "TG_Level_Stage_2",
    "Tumor_Size",
    "Time_Between_Stages",
    "Radioactive_Iodine_Dose",
]

# MLP hyperparameters (Table 1)
MLP_PARAMS = dict(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=500,
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

# EWMA control chart (Eqs. 8-10)
EWMA_LAMBDA_INITIAL = 0.15
EWMA_L_INITIAL = 3.5

# Genetic algorithm (Section 2.6)
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
