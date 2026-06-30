"""
Model Wrappers
Adapters for existing models (readmission predictor, credit scoring, cancer detection).
Provides a unified interface for training, prediction, and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from loguru import logger
from typing import Tuple, Dict, Callable
import warnings
warnings.filterwarnings("ignore")


# --- Synthetic Data Generators ---

def generate_readmission_data(n_samples: int = 5000, drift: bool = False) -> pd.DataFrame:
    """
    Generate synthetic patient readmission data.
    Simulates features from the readmission-predictor project.
    """
    np.random.seed(42 if not drift else 99)

    data = pd.DataFrame({
        "age": np.random.normal(65 if not drift else 55, 12, n_samples).clip(18, 100),
        "length_of_stay": np.random.exponential(5, n_samples).clip(1, 30),
        "num_diagnoses": np.random.poisson(5, n_samples).clip(1, 20),
        "num_medications": np.random.poisson(8, n_samples).clip(1, 30),
        "num_procedures": np.random.poisson(2, n_samples).clip(0, 10),
        "num_admissions_prev_year": np.random.poisson(1, n_samples).clip(0, 10),
        "num_emergency_visits": np.random.poisson(1, n_samples).clip(0, 10),
        "discharge_disposition": np.random.choice(["home", "snf", "home_health", "rehab"], n_samples),
        "admission_source": np.random.choice(["emergency", "referral", "transfer", "elective"], n_samples),
        "diabetes": np.random.binomial(1, 0.3, n_samples),
        "hypertension": np.random.binomial(1, 0.4, n_samples),
        "heart_failure": np.random.binomial(1, 0.2, n_samples),
    })

    # Target: probability of readmission within 30 days
    risk = (
        0.03 * data["age"] / 10
        + 0.05 * data["length_of_stay"]
        + 0.06 * data["num_diagnoses"]
        + 0.03 * data["num_medications"]
        + 0.10 * data["num_admissions_prev_year"]
        + 0.15 * data["num_emergency_visits"]
        + 0.25 * data["heart_failure"]
        + 0.10 * data["diabetes"]
        + 0.08 * data["hypertension"]
    )

    # Add drift effect: younger patients, more emergencies
    if drift:
        risk += 0.2 * data["num_emergency_visits"]

    prob = 1 / (1 + np.exp(-risk + 4))
    data["readmitted_30d"] = np.random.binomial(1, prob)
    return data


def generate_credit_data(n_samples: int = 5000, drift: bool = False) -> pd.DataFrame:
    """
    Generate synthetic credit scoring data.
    Simulates features from the credit-scoring-xai project.
    """
    np.random.seed(42 if not drift else 99)

    data = pd.DataFrame({
        "age": np.random.normal(40 if not drift else 50, 12, n_samples).clip(18, 90),
        "income": np.random.lognormal(10.5, 0.5, n_samples).clip(1000, 500000),
        "loan_amount": np.random.lognormal(10, 0.7, n_samples).clip(500, 200000),
        "credit_score": np.random.normal(650 if not drift else 580, 80, n_samples).clip(300, 850),
        "debt_to_income": np.random.beta(2, 5, n_samples),
        "num_credit_lines": np.random.poisson(5, n_samples).clip(0, 30),
        "num_late_payments": np.random.poisson(1, n_samples).clip(0, 20),
        "num_derogatory": np.random.poisson(0.3, n_samples).clip(0, 10),
        "employment_years": np.random.exponential(5, n_samples).clip(0, 40),
        "home_ownership": np.random.choice(["rent", "own", "mortgage"], n_samples),
        "loan_purpose": np.random.choice(["debt_consolidation", "home_improvement", "major_purchase", "other"], n_samples),
    })

    # Target: probability of default
    risk = (
        -0.01 * data["credit_score"] / 10
        + 0.5 * data["debt_to_income"]
        + 0.1 * data["num_late_payments"]
        + 0.3 * data["num_derogatory"]
        - 0.02 * data["employment_years"]
        + 0.0001 * data["loan_amount"]
    )

    if drift:
        risk += 0.3 * data["debt_to_income"]

    prob = 1 / (1 + np.exp(-risk - 1))
    data["default"] = np.random.binomial(1, prob)
    return data


# --- Training Functions ---

def train_readmission_model(train_data: pd.DataFrame, val_data: pd.DataFrame) -> Tuple[object, Dict, Dict]:
    """
    Train XGBoost readmission model.
    Returns (model, metrics, params).
    """
    # Preprocess
    categorical = ["discharge_disposition", "admission_source"]
    numerical = ["age", "length_of_stay", "num_diagnoses", "num_medications",
                 "num_procedures", "num_admissions_prev_year", "num_emergency_visits",
                 "diabetes", "hypertension", "heart_failure"]

    # Encode categorical (use .copy() to avoid SettingWithCopyWarning)
    train_data = train_data.copy()
    val_data = val_data.copy()
    for col in categorical:
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col].astype(str))
        val_data[col] = le.transform(val_data[col].astype(str))

    features = numerical + categorical
    X_train = train_data[features].values
    y_train = train_data["readmitted_30d"].values
    X_val = val_data[features].values
    y_val = val_data["readmitted_30d"].values

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "auc",
    }

    model = XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "auc": float(roc_auc_score(y_val, y_prob)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
    }

    logger.info(f"Readmission model trained: AUC={metrics['auc']:.4f}")
    return model, metrics, params


def train_credit_model(train_data: pd.DataFrame, val_data: pd.DataFrame) -> Tuple[object, Dict, Dict]:
    """
    Train XGBoost credit scoring model.
    Returns (model, metrics, params).
    """
    categorical = ["home_ownership", "loan_purpose"]
    numerical = ["age", "income", "loan_amount", "credit_score", "debt_to_income",
                 "num_credit_lines", "num_late_payments", "num_derogatory", "employment_years"]

    # Encode categorical (use .copy() to avoid SettingWithCopyWarning)
    train_data = train_data.copy()
    val_data = val_data.copy()
    for col in categorical:
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col].astype(str))
        val_data[col] = le.transform(val_data[col].astype(str))

    features = numerical + categorical
    X_train = train_data[features].values
    y_train = train_data["default"].values
    X_val = val_data[features].values
    y_val = val_data["default"].values

    params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "auc",
    }

    model = XGBClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "auc": float(roc_auc_score(y_val, y_prob)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
    }

    logger.info(f"Credit model trained: AUC={metrics['auc']:.4f}")
    return model, metrics, params


# --- Model Registry ---
MODEL_REGISTRY: Dict[str, Dict] = {
    "readmission_predictor": {
        "train_fn": train_readmission_model,
        "data_fn": generate_readmission_data,
        "target_col": "readmitted_30d",
        "model_type": "xgboost",
    },
    "credit_scorer": {
        "train_fn": train_credit_model,
        "data_fn": generate_credit_data,
        "target_col": "default",
        "model_type": "xgboost",
    },
}


def get_model_info(model_name: str) -> Dict:
    """Get model configuration from registry."""
    return MODEL_REGISTRY.get(model_name)