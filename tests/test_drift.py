"""
Test Drift Detection
Validates PSI calculation, feature drift, target drift, and prediction drift.
"""

import os

import numpy as np
import pandas as pd


class TestPSICalculation:
    """Test Population Stability Index calculation."""

    def test_no_drift(self, drift_detector):
        """Same distributions should have PSI close to 0."""
        data = np.random.normal(50, 10, 10000)
        psi = drift_detector.calculate_psi(data, data)
        assert psi < 0.01, f"PSI should be ~0 for identical data, got {psi}"

    def test_moderate_drift(self, drift_detector):
        """Shifted mean should produce moderate PSI."""
        ref = np.random.normal(50, 10, 10000)
        cur = np.random.normal(55, 10, 10000)
        psi = drift_detector.calculate_psi(ref, cur)
        assert 0.05 < psi < 0.5, f"Expected moderate PSI, got {psi}"

    def test_significant_drift(self, drift_detector):
        """Large shift should produce high PSI."""
        ref = np.random.normal(50, 10, 10000)
        cur = np.random.normal(70, 15, 10000)
        psi = drift_detector.calculate_psi(ref, cur)
        assert psi > 0.2, f"Expected high PSI, got {psi}"

    def test_constant_data(self, drift_detector):
        """Constant data should return 0 PSI."""
        ref = np.ones(100) * 5
        cur = np.ones(100) * 5
        psi = drift_detector.calculate_psi(ref, cur)
        assert psi == 0.0


class TestFeatureDrift:
    """Test feature-level drift detection."""

    def test_detect_feature_drift_no_drift(self, drift_detector):
        """No drift when data is from same distribution."""
        ref = pd.DataFrame({"a": np.random.normal(50, 10, 1000), "b": np.random.normal(30, 5, 1000)})
        cur = pd.DataFrame({"a": np.random.normal(50, 10, 1000), "b": np.random.normal(30, 5, 1000)})
        result = drift_detector.detect_feature_drift(ref, cur)
        assert not result["drift_detected"]

    def test_detect_feature_drift_with_drift(self, drift_detector):
        """Detect drift when data distribution changes."""
        ref = pd.DataFrame({"a": np.random.normal(50, 10, 1000), "b": np.random.normal(30, 5, 1000)})
        cur = pd.DataFrame({"a": np.random.normal(80, 20, 1000), "b": np.random.normal(30, 5, 1000)})
        result = drift_detector.detect_feature_drift(ref, cur)
        assert result["drift_detected"]
        assert "a" in result["drifted_features"]
        assert "b" not in result["drifted_features"]

    def test_categorical_drift(self, drift_detector):
        """Detect drift in categorical features."""
        ref = pd.DataFrame({"cat": np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2])})
        cur = pd.DataFrame({"cat": np.random.choice(["A", "B", "C"], 1000, p=[0.1, 0.3, 0.6])})
        result = drift_detector.detect_feature_drift(ref, cur)
        assert result["drift_detected"]
        assert "cat" in result["drifted_features"]

    def test_report_generated(self, drift_detector):
        """Verify drift report is saved to disk."""
        ref = pd.DataFrame({"a": np.random.normal(50, 10, 500)})
        cur = pd.DataFrame({"a": np.random.normal(80, 20, 500)})
        result = drift_detector.detect_feature_drift(ref, cur)
        assert "report_path" in result
        assert os.path.exists(result["report_path"])


class TestTargetDrift:
    """Test target (concept) drift detection."""

    def test_no_target_drift(self, drift_detector):
        """Same target distribution should show no drift."""
        ref = np.random.binomial(1, 0.3, 1000)
        cur = np.random.binomial(1, 0.3, 1000)
        result = drift_detector.detect_target_drift(ref, cur)
        assert not result["drifted"]

    def test_target_drift_detected(self, drift_detector):
        """Changed target distribution should show drift."""
        ref = np.random.binomial(1, 0.2, 5000)
        cur = np.random.binomial(1, 0.6, 5000)
        result = drift_detector.detect_target_drift(ref, cur)
        assert result["drifted"]


class TestPredictionDrift:
    """Test prediction distribution drift."""

    def test_no_prediction_drift(self, drift_detector):
        ref = np.random.normal(0.5, 0.1, 5000)
        cur = np.random.normal(0.5, 0.1, 5000)
        result = drift_detector.detect_prediction_drift(ref, cur)
        assert not result["drifted"]

    def test_prediction_drift(self, drift_detector):
        ref = np.random.normal(0.3, 0.1, 5000)
        cur = np.random.normal(0.7, 0.15, 5000)
        result = drift_detector.detect_prediction_drift(ref, cur)
        assert result["drifted"]
