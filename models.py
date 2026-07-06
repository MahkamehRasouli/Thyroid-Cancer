"""
models.py
---------
Implements Section 2.4 "Risk Prediction Modeling" and reproduces the
comparison in Table 4: Logistic Regression, Random Forest, SVM, KNN, DNN,
and MLP, each evaluated on Accuracy, k-fold CV Accuracy, AUC, and F1 score.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from config import (
    CV_FOLDS,
    DNN_PARAMS,
    MLP_PARAMS,
    RANDOM_STATE,
    TEST_SIZE,
)


def get_candidate_models() -> dict:
    """Return the six models compared in the manuscript's Table 4."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "DNN": MLPClassifier(**DNN_PARAMS),
        "MLP": MLPClassifier(**MLP_PARAMS),
    }


def evaluate_model(model, X, y, X_train, X_test, y_train, y_test) -> dict:
    """Fit on the train split and compute the four metrics reported in Table 4."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    cv_scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring="accuracy")

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Cross-Validation Accuracy": cv_scores.mean(),
        "AUC": roc_auc_score(y_test, y_score),
        "F1 Score": f1_score(y_test, y_pred),
    }


def compare_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Train/evaluate all candidate models and return a Table-4-style comparison."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = {}
    fitted_models = {}
    for name, model in get_candidate_models().items():
        results[name] = evaluate_model(model, X, y, X_train, X_test, y_train, y_test)
        fitted_models[name] = model

    results_df = pd.DataFrame(results).T
    results_df.index.name = "Model"
    return results_df, fitted_models


def predict_risk_scores(model, X: pd.DataFrame) -> pd.Series:
    """Generate patient-level predicted risk scores (probability of the positive class).

    These are the scores fed into the EWMA control charts in Section 2.5.
    """
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)
    return pd.Series(scores, index=X.index, name="Risk_Score")
