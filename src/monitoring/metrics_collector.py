"""
Monitoring & Alerting
Prometheus metrics collection and alerting for model performance.
"""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from datetime import datetime
from typing import Dict
from loguru import logger
import json

from src.config import settings


# Custom Prometheus metrics
REGISTRY = CollectorRegistry()

# Prediction metrics
PREDICTION_COUNT = Counter(
    "mlops_predictions_total",
    "Total predictions served",
    ["model_name", "model_version", "stage"],
    registry=REGISTRY,
)

PREDICTION_LATENCY = Histogram(
    "mlops_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model_name"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=REGISTRY,
)

# Model metrics
MODEL_ACCURACY = Gauge(
    "mlops_model_accuracy",
    "Current model accuracy",
    ["model_name", "version"],
    registry=REGISTRY,
)

MODEL_AUC = Gauge(
    "mlops_model_auc",
    "Current model AUC score",
    ["model_name", "version"],
    registry=REGISTRY,
)

# Drift metrics
DRIFT_PSI = Gauge(
    "mlops_drift_psi",
    "Population Stability Index per feature",
    ["model_name", "feature"],
    registry=REGISTRY,
)

DRIFT_DETECTED = Gauge(
    "mlops_drift_detected_total",
    "Total drift detection runs that found drift",
    ["model_name"],
    registry=REGISTRY,
)

# Retraining metrics
RETRAIN_COUNT = Counter(
    "mlops_retrain_total",
    "Total retraining executions",
    ["model_name", "status"],
    registry=REGISTRY,
)

RETRAIN_DURATION = Histogram(
    "mlops_retrain_duration_seconds",
    "Retraining pipeline duration",
    ["model_name"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800],
    registry=REGISTRY,
)


class MetricsCollector:
    """Collect and export custom metrics for Prometheus."""

    @staticmethod
    def record_prediction(model_name: str, version: str, stage: str, latency: float):
        """Record a prediction event."""
        PREDICTION_COUNT.labels(model_name=model_name, model_version=version, stage=stage).inc()
        PREDICTION_LATENCY.labels(model_name=model_name).observe(latency)

    @staticmethod
    def update_model_metrics(model_name: str, version: str, metrics: Dict[str, float]):
        """Update model performance gauges."""
        if "accuracy" in metrics:
            MODEL_ACCURACY.labels(model_name=model_name, version=version).set(metrics["accuracy"])
        if "auc" in metrics:
            MODEL_AUC.labels(model_name=model_name, version=version).set(metrics["auc"])

    @staticmethod
    def record_drift(model_name: str, feature: str, psi: float, drifted: bool):
        """Record drift detection results."""
        DRIFT_PSI.labels(model_name=model_name, feature=feature).set(psi)
        if drifted:
            DRIFT_DETECTED.labels(model_name=model_name).inc()

    @staticmethod
    def record_retrain(model_name: str, status: str, duration: float):
        """Record retraining pipeline execution."""
        RETRAIN_COUNT.labels(model_name=model_name, status=status).inc()
        RETRAIN_DURATION.labels(model_name=model_name).observe(duration)

    @staticmethod
    def export() -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(REGISTRY).decode("utf-8")


class AlertManager:
    """Alert management for drift and performance degradation."""

    def __init__(self):
        self.alerts_dir = settings.data_dir / "alerts"
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        self.active_alerts: Dict[str, dict] = {}

    def raise_alert(
        self,
        alert_id: str,
        severity: str,
        title: str,
        description: str,
        model_name: str = None,
        metadata: dict = None,
    ) -> dict:
        """Raise a new alert."""
        alert = {
            "id": alert_id,
            "severity": severity,
            "title": title,
            "description": description,
            "model_name": model_name,
            "raised_at": datetime.now().isoformat(),
            "status": "active",
            "metadata": metadata or {},
        }
        self.active_alerts[alert_id] = alert

        # Log
        if severity == "critical":
            logger.error(f"[ALERT CRITICAL] {title}: {description}")
        elif severity == "warning":
            logger.warning(f"[ALERT WARNING] {title}: {description}")
        else:
            logger.info(f"[ALERT INFO] {title}: {description}")

        # Persist
        alert_path = self.alerts_dir / f"{alert_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(alert_path, "w") as f:
            json.dump(alert, f, indent=2)

        return alert

    def resolve_alert(self, alert_id: str) -> dict:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]["status"] = "resolved"
            self.active_alerts[alert_id]["resolved_at"] = datetime.now().isoformat()
            logger.info(f"Alert resolved: {alert_id}")
            return self.active_alerts.pop(alert_id)
        return {"error": f"Alert {alert_id} not found"}

    def list_alerts(self, include_resolved: bool = False) -> list:
        """List active (and optionally resolved) alerts."""
        alerts = list(self.active_alerts.values())
        if include_resolved:
            # Also load historical alerts from disk
            for alert_file in sorted(self.alerts_dir.glob("*.json"))[-10:]:
                with open(alert_file) as f:
                    alert = json.load(f)
                    if alert["id"] not in self.active_alerts:
                        alerts.append(alert)
        return alerts

    def check_and_alert(
        self,
        model_name: str,
        drift_report: dict,
        performance_metrics: dict = None,
        baseline_metrics: dict = None,
    ) -> list:
        """Evaluate conditions and raise alerts if needed."""
        new_alerts = []

        # Drift alert
        if drift_report.get("drift_detected", False):
            drifted = drift_report.get("drifted_features", [])
            alert = self.raise_alert(
                alert_id=f"drift_{model_name}_{datetime.now().strftime('%Y%m%d')}",
                severity="warning",
                title=f"Data drift detected for {model_name}",
                description=f"Drift detected in {len(drifted)} features: {', '.join(drifted[:5])}",
                model_name=model_name,
                metadata={"drifted_features": drifted},
            )
            new_alerts.append(alert)

        # Performance degradation alert
        if performance_metrics and baseline_metrics:
            for metric, baseline_val in baseline_metrics.items():
                current_val = performance_metrics.get(metric, 0)
                degradation = baseline_val - current_val
                if degradation > settings.retrain_performance_threshold:
                    alert = self.raise_alert(
                        alert_id=f"perf_{model_name}_{metric}_{datetime.now().strftime('%Y%m%d')}",
                        severity="critical",
                        title=f"Performance degradation: {model_name}.{metric}",
                        description=f"{metric} dropped from {baseline_val:.4f} to {current_val:.4f} (delta={degradation:.4f})",
                        model_name=model_name,
                        metadata={
                            "metric": metric,
                            "baseline": baseline_val,
                            "current": current_val,
                            "degradation": degradation,
                        },
                    )
                    new_alerts.append(alert)

        return new_alerts
