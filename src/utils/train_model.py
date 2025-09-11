import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train(input_csv: str):
    """
    Train a leakage-free pipeline with preprocessing and calibrated probabilities.

    Returns
    -------
    df : pd.DataFrame
        Cleaned full dataset (features + target) for UI visualizations.
    model : CalibratedClassifierCV
        Calibrated pipeline (preprocessing + logistic regression).
    metrics : dict
        Evaluation metrics on the held-out test set.
    X_test : pd.DataFrame
        Test features (raw schema pre-preprocessing).
    y_test : pd.Series
        Test labels.
    y_proba_test : np.ndarray
        Predicted positive-class probabilities for X_test.
    feature_schema : dict
        Dict describing numeric and categorical feature names (raw inputs expected by the model).
    """

    # Column names (UCI Cleveland)
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
    ]
    df = pd.read_csv(input_csv, header=None, names=column_names)

    # Clean and coerce
    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Binary target
    df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)

    # Raw features (no manual one-hot or scaling here)
    X = df.drop('num', axis=1)
    y = df['num']

    # Define schema for preprocessing
    numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    feature_schema = {
        "numeric": numeric_features,
        "categorical": categorical_features,
    }

    # Preprocessor and model
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    base_clf = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ],
        memory=None,
    )

    calibrated_model = CalibratedClassifierCV(
        base_estimator=base_clf,
        method="isotonic",
        cv=5,
    )

    # Split without leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    calibrated_model.fit(X_train, y_train)

    # Evaluate
    y_pred = calibrated_model.predict(X_test)
    y_proba = calibrated_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
    }

    return df, calibrated_model, metrics, X_test, y_test, y_proba, feature_schema