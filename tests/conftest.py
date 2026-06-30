"""
Pytest Configuration and Shared Fixtures
Auto-loaded by pytest for all test modules.
"""

import os
import sys
import numpy as np
import pytest

# Add project root to path (replaces per-file sys.path.insert)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# --- Fixtures: Data Generators ---


@pytest.fixture
def reference_data():
    """Generate reference (baseline) patient readmission data."""
    from src.models.model_wrappers import generate_readmission_data

    return generate_readmission_data(n_samples=1000, drift=False)


@pytest.fixture
def drifted_data():
    """Generate drifted patient readmission data."""
    from src.models.model_wrappers import generate_readmission_data

    return generate_readmission_data(n_samples=1000, drift=True)


@pytest.fixture
def credit_reference_data():
    """Generate reference credit scoring data."""
    from src.models.model_wrappers import generate_credit_data

    return generate_credit_data(n_samples=1000, drift=False)


@pytest.fixture
def credit_drifted_data():
    """Generate drifted credit scoring data."""
    from src.models.model_wrappers import generate_credit_data

    return generate_credit_data(n_samples=1000, drift=True)


# --- Fixtures: Module Instances ---


@pytest.fixture
def drift_detector():
    """DriftDetector instance."""
    from src.drift.drift_detector import DriftDetector

    return DriftDetector()


@pytest.fixture
def ab_test_manager():
    """ABTestManager instance."""
    from src.ab_testing.ab_test_manager import ABTestManager

    return ABTestManager()


@pytest.fixture
def metrics_collector():
    """MetricsCollector instance."""
    from src.monitoring.metrics_collector import MetricsCollector

    return MetricsCollector()


@pytest.fixture
def alert_manager():
    """AlertManager instance."""
    from src.monitoring.metrics_collector import AlertManager

    return AlertManager()


# --- Fixtures: API Client ---


@pytest.fixture
def client():
    """FastAPI TestClient."""
    from fastapi.testclient import TestClient
    from src.serving.api import app

    return TestClient(app)


# --- Fixtures: Synthetic Arrays ---


@pytest.fixture
def no_drift_arrays():
    """Two arrays from the same distribution (no drift)."""
    np.random.seed(42)
    ref = np.random.normal(50, 10, 5000)
    cur = np.random.normal(50, 10, 5000)
    return ref, cur


@pytest.fixture
def drift_arrays():
    """Two arrays from different distributions (significant drift)."""
    np.random.seed(42)
    ref = np.random.normal(50, 10, 5000)
    cur = np.random.normal(80, 20, 5000)
    return ref, cur


# --- Fixtures: Cleanup ---


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Clean up generated test data after each test module."""
    yield
    # Cleanup is handled after yield (teardown)
    from pathlib import Path

    base = Path(__file__).parent.parent / "data"
    for subdir in ["ab_tests", "alerts", "reports", "artifacts"]:
        d = base / subdir
        if d.exists():
            for f in d.glob("*"):
                try:
                    f.unlink()
                except OSError:
                    pass
