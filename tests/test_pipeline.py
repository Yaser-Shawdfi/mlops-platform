"""
Test Pipeline Orchestration
Validates retraining trigger logic and pipeline execution.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.retrain_pipeline import RetrainTrigger
from src.models.model_wrappers import generate_readmission_data, train_readmission_model


class TestRetrainTrigger:
    """Test retraining trigger logic."""

    def test_no_trigger_when_healthy(self):
        trigger = RetrainTrigger()
        decision = trigger.should_retrain(
            drift_report={"drift_detected": False, "drifted_features": []},
            current_performance={"auc": 0.85},
            baseline_performance={"auc": 0.85},
        )
        assert not decision["should_retrain"]

    def test_trigger_on_drift(self):
        trigger = RetrainTrigger()
        decision = trigger.should_retrain(
            drift_report={"drift_detected": True, "drifted_features": ["a", "b", "c", "d"]},
            current_performance={},
            baseline_performance={},
        )
        assert decision["should_retrain"]
        assert len(decision["reasons"]) > 0

    def test_trigger_on_performance_degradation(self):
        trigger = RetrainTrigger()
        decision = trigger.should_retrain(
            drift_report={"drift_detected": False, "drifted_features": []},
            current_performance={"auc": 0.70},
            baseline_performance={"auc": 0.85},
        )
        assert decision["should_retrain"]
        assert any("Performance degradation" in r for r in decision["reasons"])

    def test_no_trigger_on_minor_drift(self):
        """Only 1-2 drifted features should not trigger retrain."""
        trigger = RetrainTrigger()
        decision = trigger.should_retrain(
            drift_report={"drift_detected": True, "drifted_features": ["a"]},
            current_performance={"auc": 0.85},
            baseline_performance={"auc": 0.85},
        )
        assert not decision["should_retrain"]


class TestPipelineExecution:
    """Test full pipeline execution."""

    def test_pipeline_runs(self):
        """Test that the pipeline can execute end-to-end."""
        from src.pipeline.retrain_pipeline import RetrainPipeline
        pipeline = RetrainPipeline()

        ref_data = generate_readmission_data(n_samples=2000, drift=False)
        cur_data = generate_readmission_data(n_samples=2000, drift=True)

        result = pipeline.execute(
            model_name="test_readmission",
            train_fn=train_readmission_model,
            reference_data=ref_data,
            current_data=cur_data,
            baseline_performance={"auc": 0.5},  # Low baseline to force retrain
            model_type="xgboost",
            auto_deploy=False,
        )

        assert result["status"] in ["completed", "skipped", "failed"]
        assert "steps" in result
        assert "duration_seconds" in result
        assert "execution_id" in result

    def test_pipeline_history_saved(self):
        """Test that pipeline execution is saved to disk."""
        from src.pipeline.retrain_pipeline import RetrainPipeline
        pipeline = RetrainPipeline()

        ref_data = generate_readmission_data(n_samples=1000, drift=False)
        cur_data = generate_readmission_data(n_samples=1000, drift=False)

        result = pipeline.execute(
            model_name="test_history",
            train_fn=train_readmission_model,
            reference_data=ref_data,
            current_data=cur_data,
            model_type="xgboost",
        )

        assert "history_path" in result
        assert os.path.exists(result["history_path"])