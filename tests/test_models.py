"""
Test Model Wrappers
Validates synthetic data generation and model training.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.model_wrappers import (
    generate_readmission_data,
    generate_credit_data,
    train_readmission_model,
    train_credit_model,
)
from sklearn.model_selection import train_test_split


class TestReadmissionData:
    """Test readmission data generation."""

    def test_generates_correct_shape(self):
        data = generate_readmission_data(n_samples=1000)
        assert data.shape[0] == 1000
        assert "readmitted_30d" in data.columns

    def test_target_is_binary(self):
        data = generate_readmission_data(n_samples=500)
        assert set(data["readmitted_30d"].unique()).issubset({0, 1})

    def test_drift_changes_distribution(self):
        ref = generate_readmission_data(n_samples=2000, drift=False)
        cur = generate_readmission_data(n_samples=2000, drift=True)
        # Drifted data should have different age mean
        assert abs(ref["age"].mean() - cur["age"].mean()) > 5

    def test_no_nan_values(self):
        data = generate_readmission_data(n_samples=500)
        assert not data.isnull().any().any()


class TestCreditData:
    """Test credit data generation."""

    def test_generates_correct_shape(self):
        data = generate_credit_data(n_samples=1000)
        assert data.shape[0] == 1000
        assert "default" in data.columns

    def test_target_is_binary(self):
        data = generate_credit_data(n_samples=500)
        assert set(data["default"].unique()).issubset({0, 1})

    def test_credit_score_in_range(self):
        data = generate_credit_data(n_samples=500)
        assert data["credit_score"].min() >= 300
        assert data["credit_score"].max() <= 850


class TestModelTraining:
    """Test model training functions."""

    def test_readmission_model_trains(self):
        data = generate_readmission_data(n_samples=2000)
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)
        model, metrics, params = train_readmission_model(train_df.copy(), val_df.copy())

        assert model is not None
        assert "auc" in metrics
        assert "accuracy" in metrics
        assert metrics["auc"] > 0.5  # Better than random
        assert "n_estimators" in params

    def test_credit_model_trains(self):
        data = generate_credit_data(n_samples=2000)
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)
        model, metrics, params = train_credit_model(train_df.copy(), val_df.copy())

        assert model is not None
        assert "auc" in metrics
        assert "accuracy" in metrics
        assert metrics["auc"] > 0.5
        assert "n_estimators" in params

    def test_readmission_metrics_reasonable(self):
        data = generate_readmission_data(n_samples=5000)
        train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)
        model, metrics, _ = train_readmission_model(train_df.copy(), val_df.copy())

        assert 0.45 < metrics["accuracy"] < 1.0
        assert 0.45 < metrics["auc"] < 1.0
        assert 0 <= metrics["precision"] <= 1.0
        assert 0 <= metrics["recall"] <= 1.0