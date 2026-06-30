"""
Test Monitoring & Alerting
Validates metrics collection and alert management.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.monitoring.metrics_collector import MetricsCollector, AlertManager


class TestMetricsCollector:
    """Test Prometheus metrics collection."""

    def test_record_prediction(self):
        collector = MetricsCollector()
        collector.record_prediction("test_model", "1", "Production", 0.05)
        # Should not raise

    def test_update_model_metrics(self):
        collector = MetricsCollector()
        collector.update_model_metrics("test_model", "1", {"accuracy": 0.85, "auc": 0.92})
        # Should not raise

    def test_record_drift(self):
        collector = MetricsCollector()
        collector.record_drift("test_model", "feature_a", 0.35, True)
        collector.record_drift("test_model", "feature_b", 0.05, False)

    def test_record_retrain(self):
        collector = MetricsCollector()
        collector.record_retrain("test_model", "completed", 45.5)

    def test_export(self):
        collector = MetricsCollector()
        collector.record_prediction("test_model", "1", "Production", 0.05)
        exported = collector.export()
        assert isinstance(exported, str)
        assert "mlops_predictions_total" in exported


class TestAlertManager:
    """Test alert management."""

    def test_raise_alert(self):
        mgr = AlertManager()
        alert = mgr.raise_alert(
            alert_id="test_alert_1",
            severity="warning",
            title="Test Alert",
            description="Testing alert system",
            model_name="test_model",
        )
        assert alert["id"] == "test_alert_1"
        assert alert["severity"] == "warning"
        assert alert["status"] == "active"

    def test_resolve_alert(self):
        mgr = AlertManager()
        mgr.raise_alert("test_resolve", "critical", "Test", "Test desc")
        result = mgr.resolve_alert("test_resolve")
        assert result["status"] == "resolved"

    def test_list_alerts(self):
        mgr = AlertManager()
        mgr.raise_alert("test_list_1", "warning", "Alert 1", "Desc 1")
        mgr.raise_alert("test_list_2", "critical", "Alert 2", "Desc 2")
        alerts = mgr.list_alerts()
        assert len(alerts) >= 2

    def test_check_and_alert_drift(self):
        mgr = AlertManager()
        drift_report = {
            "drift_detected": True,
            "drifted_features": ["age", "income", "credit_score"],
        }
        alerts = mgr.check_and_alert("test_model", drift_report)
        assert len(alerts) >= 1
        assert alerts[0]["severity"] == "warning"

    def test_check_and_alert_performance(self):
        mgr = AlertManager()
        alerts = mgr.check_and_alert(
            "test_model",
            drift_report={"drift_detected": False, "drifted_features": []},
            performance_metrics={"auc": 0.70},
            baseline_metrics={"auc": 0.85},
        )
        assert len(alerts) >= 1
        assert alerts[0]["severity"] == "critical"

    def test_check_no_alert_when_healthy(self):
        mgr = AlertManager()
        alerts = mgr.check_and_alert(
            "test_model",
            drift_report={"drift_detected": False, "drifted_features": []},
            performance_metrics={"auc": 0.85},
            baseline_metrics={"auc": 0.85},
        )
        assert len(alerts) == 0
