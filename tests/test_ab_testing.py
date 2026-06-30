"""
Test A/B Testing Framework
Validates test creation, routing, result recording, and statistical evaluation.
"""

import pytest
import numpy as np
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ab_testing.ab_test_manager import ABTestManager
from src.config import settings


class TestABTestCreation:
    """Test A/B test creation and configuration."""

    def test_create_test(self):
        mgr = ABTestManager()
        test = mgr.create_test(
            test_name="test_create_basic",
            model_name="test_model",
            version_a="1",
            version_b="2",
            traffic_split=0.5,
        )
        assert test["test_name"] == "test_create_basic"
        assert test["version_a"] == "1"
        assert test["version_b"] == "2"
        assert test["status"] == "running"

    def test_create_test_custom_split(self):
        mgr = ABTestManager()
        test = mgr.create_test(
            test_name="test_create_split",
            model_name="test_model",
            version_a="1",
            version_b="2",
            traffic_split=0.3,
        )
        assert test["traffic_split"] == 0.3

    def test_test_config_saved(self):
        mgr = ABTestManager()
        mgr.create_test(
            test_name="test_saved_config",
            model_name="test_model",
            version_a="1",
            version_b="2",
        )
        config_path = settings.data_dir / "ab_tests" / "test_saved_config_config.json"
        assert config_path.exists()


class TestRouting:
    """Test traffic routing."""

    def test_route_returns_valid_version(self):
        mgr = ABTestManager()
        mgr.create_test(
            test_name="test_routing",
            model_name="test_model",
            version_a="1",
            version_b="2",
            traffic_split=0.5,
        )
        for _ in range(100):
            version = mgr.route_request("test_routing")
            assert version in ["A", "B"]

    def test_route_nonexistent_test_returns_a(self):
        mgr = ABTestManager()
        version = mgr.route_request("nonexistent_test")
        assert version == "A"


class TestResultRecording:
    """Test result recording."""

    def test_record_result(self):
        mgr = ABTestManager()
        mgr.create_test(
            test_name="test_record",
            model_name="test_model",
            version_a="1",
            version_b="2",
            min_sample_size=10,
        )
        result = mgr.record_result("test_record", "A", 0.7, actual=1)
        assert result["recorded"] is True
        assert result["total_a"] == 1

    def test_record_multiple_results(self):
        mgr = ABTestManager()
        mgr.create_test(
            test_name="test_multi_record",
            model_name="test_model",
            version_a="1",
            version_b="2",
            min_sample_size=10,
        )
        for i in range(20):
            mgr.record_result("test_multi_record", "A", 0.7, actual=1 if i % 2 == 0 else 0)
            mgr.record_result("test_multi_record", "B", 0.8, actual=1 if i % 3 == 0 else 0)
        result = mgr.record_result("test_multi_record", "A", 0.7, actual=1)
        assert result["total_a"] == 21


class TestEvaluation:
    """Test statistical evaluation."""

    def test_insufficient_data(self):
        mgr = ABTestManager()
        mgr.create_test(
            test_name="test_insufficient",
            model_name="test_model",
            version_a="1",
            version_b="2",
            min_sample_size=1000,
        )
        mgr.record_result("test_insufficient", "A", 0.7, actual=1)
        result = mgr.evaluate_test("test_insufficient")
        assert result["status"] == "insufficient_data"

    def test_evaluation_with_outcomes(self):
        mgr = ABTestManager()
        mgr.create_test(
            test_name="test_eval_outcomes",
            model_name="test_model",
            version_a="1",
            version_b="2",
            min_sample_size=50,
        )
        np.random.seed(42)
        # Version B is better
        for _ in range(100):
            mgr.record_result("test_eval_outcomes", "A", np.random.beta(2, 5), actual=np.random.binomial(1, 0.3))
            mgr.record_result("test_eval_outcomes", "B", np.random.beta(3, 4), actual=np.random.binomial(1, 0.4))

        result = mgr.evaluate_test("test_eval_outcomes")
        assert result["status"] == "completed"
        assert "accuracy_a" in result
        assert "accuracy_b" in result
        assert "p_value" in result
        assert "recommendation" in result


class TestListAndStop:
    """Test listing and stopping tests."""

    def test_list_tests(self):
        mgr = ABTestManager()
        mgr.create_test("test_list_1", "model", "1", "2")
        tests = mgr.list_tests()
        assert any(t["test_name"] == "test_list_1" for t in tests)

    def test_stop_test(self):
        mgr = ABTestManager()
        mgr.create_test("test_stop", "model", "1", "2")
        result = mgr.stop_test("test_stop")
        assert result["status"] == "stopped"