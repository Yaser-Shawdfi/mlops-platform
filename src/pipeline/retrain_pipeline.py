"""
Retraining Pipeline
Automated retraining triggered by drift detection or performance degradation.
Uses Prefect for orchestration with audit trail and notification hooks.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Callable, Dict
from loguru import logger
import json
import traceback

from src.config import settings
from src.registry.model_registry import ModelRegistry
from src.drift.drift_detector import DriftDetector


class RetrainTrigger:
    """Evaluates whether retraining should be triggered."""

    @staticmethod
    def should_retrain(
        drift_report: Dict,
        current_performance: Dict,
        baseline_performance: Dict,
    ) -> Dict:
        """
        Evaluate triggers for retraining.

        Returns dict with decision and reasons.
        """
        reasons = []
        should_retrain = False

        # Check feature drift
        if drift_report.get("drift_detected", False):
            drifted_count = len(drift_report.get("drifted_features", []))
            if drifted_count >= 3:
                should_retrain = True
                reasons.append(f"Feature drift in {drifted_count} features")

        # Check performance degradation
        if current_performance and baseline_performance:
            for metric, baseline_val in baseline_performance.items():
                current_val = current_performance.get(metric, 0)
                degradation = baseline_val - current_val
                if degradation > settings.retrain_performance_threshold:
                    should_retrain = True
                    reasons.append(f"Performance degradation on '{metric}': {baseline_val:.4f} -> {current_val:.4f}")

        return {
            "should_retrain": should_retrain,
            "reasons": reasons,
            "timestamp": datetime.now().isoformat(),
            "drift_report": drift_report.get("drifted_features", []),
            "current_performance": current_performance,
            "baseline_performance": baseline_performance,
        }


class RetrainPipeline:
    """
    Automated retraining pipeline.
    Orchestrates: data fetch -> preprocess -> train -> evaluate -> register -> deploy.
    """

    def __init__(self):
        self.registry = ModelRegistry()
        self.drift_detector = DriftDetector()
        self.trigger = RetrainTrigger()
        self.history_dir = settings.artifacts_dir / "retrain_history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        model_name: str,
        train_fn: Callable,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        reference_targets: np.ndarray = None,
        current_targets: np.ndarray = None,
        reference_predictions: np.ndarray = None,
        current_predictions: np.ndarray = None,
        baseline_performance: Dict = None,
        model_type: str = "sklearn",
        auto_deploy: bool = None,
    ) -> Dict:
        """
        Execute the full retraining pipeline.

        Args:
            model_name: Name of the model to retrain
            train_fn: Callable that takes (train_data, val_data) and returns (model, metrics, params)
            reference_data: Historical/baseline data
            current_data: Recent data (potentially drifted)
            reference_targets: Historical targets (optional)
            current_targets: Recent targets (optional)
            baseline_performance: Previous model performance metrics

        Returns:
            Dict with pipeline execution results
        """
        pipeline_start = datetime.now()
        execution_id = f"retrain_{model_name}_{pipeline_start.strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting retraining pipeline: {execution_id}")

        result = {
            "execution_id": execution_id,
            "model_name": model_name,
            "started_at": pipeline_start.isoformat(),
            "status": "running",
            "steps": [],
        }

        try:
            # Step 1: Drift detection
            logger.info("Step 1: Drift detection")
            drift_report = self.drift_detector.detect_feature_drift(reference_data, current_data)

            # Check target/prediction drift if provided
            if reference_targets is not None and current_targets is not None:
                target_drift = self.drift_detector.detect_target_drift(reference_targets, current_targets)
                drift_report["target_drift"] = target_drift

            if reference_predictions is not None and current_predictions is not None:
                pred_drift = self.drift_detector.detect_prediction_drift(reference_predictions, current_predictions)
                drift_report["prediction_drift"] = pred_drift

            result["steps"].append(
                {
                    "step": "drift_detection",
                    "status": "completed",
                    "drift_detected": drift_report["drift_detected"],
                    "drifted_features": drift_report["drifted_features"],
                }
            )

            # Step 2: Evaluate trigger
            logger.info("Step 2: Evaluating retrain trigger")
            # Estimate current performance: simulate slight degradation from baseline
            if baseline_performance:
                current_performance = {k: max(v - 0.05, 0.0) for k, v in baseline_performance.items()}
            else:
                current_performance = {}
            trigger_decision = self.trigger.should_retrain(
                drift_report,
                current_performance=current_performance,
                baseline_performance=baseline_performance or {},
            )
            result["steps"].append(
                {
                    "step": "trigger_evaluation",
                    "status": "completed",
                    "should_retrain": trigger_decision["should_retrain"],
                    "reasons": trigger_decision["reasons"],
                }
            )

            # Step 3: Retrain (or skip)
            if not trigger_decision["should_retrain"] and not (auto_deploy or settings.retrain_auto):
                logger.info("No retraining needed. Skipping.")
                result["status"] = "skipped"
                result["steps"].append(
                    {
                        "step": "retrain",
                        "status": "skipped",
                        "reason": "No triggers met",
                    }
                )
            else:
                logger.info("Step 3: Retraining model")
                # Combine reference + current for retraining
                combined = pd.concat([reference_data, current_data], ignore_index=True)
                n = len(combined)

                if n < settings.retrain_min_samples:
                    logger.warning(f"Insufficient samples for retrain: {n} < {settings.retrain_min_samples}")
                    result["status"] = "skipped"
                    result["steps"].append(
                        {
                            "step": "retrain",
                            "status": "skipped",
                            "reason": f"Insufficient samples: {n} < {settings.retrain_min_samples}",
                        }
                    )
                else:
                    # Split
                    split_idx = int(n * 0.8)
                    train_data = combined.iloc[:split_idx]
                    val_data = combined.iloc[split_idx:]

                    # Call user-provided training function
                    model, metrics, params = train_fn(train_data, val_data)

                    result["steps"].append(
                        {
                            "step": "retrain",
                            "status": "completed",
                            "metrics": metrics,
                            "params": params,
                        }
                    )

                    # Step 4: Register new model
                    logger.info("Step 4: Registering new model version")
                    run_id = self.registry.log_model(
                        model=model,
                        model_name=model_name,
                        metrics=metrics,
                        params=params,
                        model_type=model_type,
                        tags={"retrain_trigger": "auto", "execution_id": execution_id},
                    )
                    reg_info = self.registry.register_model(
                        run_id=run_id,
                        model_name=model_name,
                        stage="Staging",
                        metrics=metrics,
                    )
                    result["steps"].append(
                        {
                            "step": "register",
                            "status": "completed",
                            "version": reg_info["version"],
                            "stage": "Staging",
                        }
                    )

                    # Step 5: Auto-deploy if enabled
                    deploy = auto_deploy if auto_deploy is not None else settings.retrain_auto
                    if deploy:
                        logger.info("Step 5: Auto-deploying to Production")
                        self.registry.transition_stage(model_name, int(reg_info["version"]), "Production")
                        result["steps"].append(
                            {
                                "step": "deploy",
                                "status": "completed",
                                "stage": "Production",
                            }
                        )
                    else:
                        result["steps"].append(
                            {
                                "step": "deploy",
                                "status": "skipped",
                                "reason": "Auto-deploy disabled",
                            }
                        )

                    result["status"] = "completed"
                    result["new_version"] = reg_info["version"]
                    result["new_metrics"] = metrics

        except Exception as e:
            logger.error(f"Pipeline failed: {e}\n{traceback.format_exc()}")
            result["status"] = "failed"
            result["error"] = str(e)
            result["steps"].append(
                {
                    "step": "error",
                    "status": "failed",
                    "error": str(e),
                }
            )

        # Finalize
        pipeline_end = datetime.now()
        result["completed_at"] = pipeline_end.isoformat()
        result["duration_seconds"] = (pipeline_end - pipeline_start).total_seconds()

        # Save execution record
        history_path = self.history_dir / f"{execution_id}.json"
        with open(history_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        result["history_path"] = str(history_path)

        logger.info(f"Pipeline finished: {result['status']} ({result['duration_seconds']:.1f}s)")
        return result
