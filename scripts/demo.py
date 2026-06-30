#!/usr/bin/env python
"""
MLOps Platform Demo
Demonstrates the full ML lifecycle: train -> register -> serve -> drift detect -> retrain -> A/B test.
Uses synthetic data mimicking the readmission-predictor and credit-scoring-xai projects.
"""

import json
import os
import sys
import warnings

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
import numpy as np

from src.config import settings
from src.models.model_wrappers import (
    generate_readmission_data,
    train_readmission_model,
)
from src.drift.drift_detector import DriftDetector
from src.monitoring.metrics_collector import MetricsCollector, AlertManager
from src.ab_testing.ab_test_manager import ABTestManager

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
)

warnings.filterwarnings("ignore")


def section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_full_lifecycle():
    """Run the full MLOps lifecycle demo."""

    section("STEP 1: Generate Reference Data (Baseline)")
    ref_data = generate_readmission_data(n_samples=5000, drift=False)
    print(f"Reference data: {ref_data.shape[0]} samples, {ref_data.shape[1]} columns")
    print(f"Target distribution: {ref_data['readmitted_30d'].value_counts().to_dict()}")
    print(f"Columns: {list(ref_data.columns)}")

    section("STEP 2: Train & Log Model to MLflow")
    from src.registry.model_registry import ModelRegistry

    registry = ModelRegistry()

    # Split data
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(ref_data, test_size=0.2, random_state=42)

    model, metrics, params = train_readmission_model(train_df.copy(), val_df.copy())
    print(f"Training metrics: {json.dumps(metrics, indent=2)}")

    # Log to MLflow
    run_id = registry.log_model(
        model=model,
        model_name="readmission_predictor",
        metrics=metrics,
        params=params,
        model_type="xgboost",
        tags={"demo": "true", "source": "mlops_platform"},
    )
    print(f"MLflow run ID: {run_id}")

    reg_info = registry.register_model(
        run_id=run_id,
        model_name="readmission_predictor",
        stage="Production",
        metrics=metrics,
    )
    print(f"Registered: {reg_info['model_name']} v{reg_info['version']} -> {reg_info['stage']}")

    section("STEP 3: Update Monitoring Metrics")
    collector = MetricsCollector()
    collector.update_model_metrics("readmission_predictor", reg_info["version"], metrics)
    print("Prometheus metrics updated (accuracy, auc)")

    section("STEP 4: Simulate Time Passing -> Generate Current Data (with drift)")
    current_data = generate_readmission_data(n_samples=3000, drift=True)
    print(f"Current data: {current_data.shape[0]} samples (with drift injected)")
    print(f"Age mean shifted: ref={ref_data['age'].mean():.1f} -> cur={current_data['age'].mean():.1f}")
    print(
        f"Emergency visits: ref={ref_data['num_emergency_visits'].mean():.1f} -> cur={current_data['num_emergency_visits'].mean():.1f}"
    )

    section("STEP 5: Detect Data Drift")
    detector = DriftDetector()
    drift_report = detector.detect_feature_drift(ref_data, current_data)
    print(f"Drift detected: {drift_report['drift_detected']}")
    print(f"Drifted features: {drift_report['drifted_features']}")
    print("\nPer-feature PSI:")
    for feat, info in drift_report["features"].items():
        status = "DRIFT" if info["drifted"] else "ok"
        print(f"  {feat:30s} PSI={info['psi']:.6f}  [{status}]")

    section("STEP 6: Evaluate Alerts")
    alert_mgr = AlertManager()
    alerts = alert_mgr.check_and_alert(
        model_name="readmission_predictor",
        drift_report=drift_report,
        performance_metrics={"auc": metrics["auc"] - 0.15},  # Simulate degradation
        baseline_metrics={"auc": metrics["auc"]},
    )
    print(f"Alerts raised: {len(alerts)}")
    for a in alerts:
        print(f"  [{a['severity'].upper()}] {a['title']}: {a['description']}")

    section("STEP 7: Automated Retraining Pipeline")
    from src.pipeline.retrain_pipeline import RetrainPipeline

    pipeline = RetrainPipeline()
    result = pipeline.execute(
        model_name="readmission_predictor",
        train_fn=train_readmission_model,
        reference_data=ref_data,
        current_data=current_data,
        baseline_performance=metrics,
        model_type="xgboost",
        auto_deploy=False,
    )
    print(f"Pipeline status: {result['status']}")
    print(f"Duration: {result['duration_seconds']:.1f}s")
    for step in result["steps"]:
        print(f"  Step: {step['step']:25s} -> {step['status']}")

    section("STEP 8: A/B Test (Old Model vs New Model)")
    ab_mgr = ABTestManager()

    # Create test
    ab_mgr.create_test(
        test_name="readmission_v1_vs_v2",
        model_name="readmission_predictor",
        version_a="1",
        version_b="2",
        traffic_split=0.5,
        primary_metric="accuracy",
        min_sample_size=200,  # Lower for demo
    )

    # Simulate A/B test data
    np.random.seed(42)
    for _ in range(500):
        version = ab_mgr.route_request("readmission_v1_vs_v2")
        # Simulate: version B is slightly better
        if version == "A":
            pred = np.random.beta(2, 5)
            actual = 1 if pred > 0.4 else 0
        else:
            pred = np.random.beta(2.5, 4.5)
            actual = 1 if pred > 0.38 else 0
        ab_mgr.record_result("readmission_v1_vs_v2", version, pred, actual)

    # Evaluate
    eval_result = ab_mgr.evaluate_test("readmission_v1_vs_v2")
    print(f"A/B Test Result: {json.dumps(eval_result, indent=2)}")

    section("DEMO COMPLETE")
    print("\nSummary:")
    print("  - Model trained and registered to MLflow")
    print(f"  - Drift detected in {len(drift_report['drifted_features'])} features")
    print(f"  - {len(alerts)} alerts raised")
    print(f"  - Retraining pipeline: {result['status']}")
    print(f"  - A/B test recommendation: {eval_result.get('recommendation', eval_result.get('status'))}")
    print(f"\n  Data directory: {settings.data_dir}")
    print(f"  Drift reports:  {settings.reports_dir / 'drift'}")
    print(f"  Retrain history: {settings.artifacts_dir / 'retrain_history'}")
    print(f"  A/B test data:   {settings.data_dir / 'ab_tests'}")
    print("\n  To serve the API:  uvicorn src.serving.api:app --reload")
    print("  API docs at:        http://localhost:8000/docs")
    print("  Metrics at:        http://localhost:8000/metrics")


if __name__ == "__main__":
    demo_full_lifecycle()
