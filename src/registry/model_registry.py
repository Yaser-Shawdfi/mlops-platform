"""
MLflow Model Registry
Manages model registration, versioning, and stage transitions.
Wraps MLflow tracking with enterprise features: audit logging,
stage transitions, model comparison, and registry queries.
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from datetime import datetime
from typing import Optional
from loguru import logger
from pathlib import Path
import pandas as pd
import json
import pickle

from src.config import settings


class ModelRegistry:
    """Enterprise MLflow model registry with audit trail."""

    def __init__(self):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        self._ensure_experiment()

    def _ensure_experiment(self):
        """Create experiment if it doesn't exist."""
        try:
            exp = self.client.get_experiment_by_name(settings.mlflow_experiment_name)
            if exp is None:
                self.client.create_experiment(settings.mlflow_experiment_name)
                logger.info(f"Created MLflow experiment: {settings.mlflow_experiment_name}")
        except Exception as e:
            logger.warning(f"MLflow connection failed (will use local fallback): {e}")

    def log_model(
        self,
        model,
        model_name: str,
        metrics: dict,
        params: dict,
        artifacts: dict = None,
        model_type: str = "sklearn",
        tags: dict = None,
    ) -> str:
        """
        Log a trained model to MLflow with full metadata.

        Args:
            model: Trained model object
            model_name: Name for the model in registry
            metrics: Dict of metrics (accuracy, auc, etc.)
            params: Dict of hyperparameters
            artifacts: Dict of extra artifacts to log
            model_type: sklearn, xgboost, or lightgbm
            tags: Dict of tags for the run

        Returns:
            Run ID
        """
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            # Log params
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            # Log tags
            default_tags = {"model_name": model_name, "model_type": model_type}
            if tags:
                default_tags.update(tags)
            mlflow.set_tags(default_tags)

            # Log model based on type
            if model_type == "xgboost":
                mlflow.xgboost.log_model(model, model_name)
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(model, model_name)
            else:
                mlflow.sklearn.log_model(model, model_name)

            # Log extra artifacts
            if artifacts:
                for name, path in artifacts.items():
                    if Path(path).exists():
                        mlflow.log_artifact(path, artifact_path=name)

            run_id = run.info.run_id
            logger.info(f"Logged model '{model_name}' to MLflow | run_id={run_id} | metrics={metrics}")
            return run_id

    def register_model(
        self,
        run_id: str,
        model_name: str,
        stage: str = "Staging",
        metrics: dict = None,
    ) -> dict:
        """
        Register a model from a run into the model registry.

        Args:
            run_id: MLflow run ID
            model_name: Registered model name
            stage: None, Staging, Production, Archived
            metrics: Metrics to include in description

        Returns:
            Dict with model version info
        """
        model_uri = f"runs:/{run_id}/{model_name}"

        # Register model
        result = mlflow.register_model(model_uri, model_name)
        version = result.version

        # Transition to stage
        if stage:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=True,
            )

        # Update description with metrics
        if metrics:
            desc = " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            self.client.update_model_version(
                name=model_name,
                version=version,
                description=f"Registered {datetime.now().isoformat()} | {desc}",
            )

        info = {
            "model_name": model_name,
            "version": version,
            "stage": stage,
            "run_id": run_id,
            "registered_at": datetime.now().isoformat(),
            "metrics": metrics or {},
        }
        logger.info(f"Registered model '{model_name}' v{version} -> {stage}")
        return info

    def get_latest_model(self, model_name: str, stage: str = "Production") -> dict:
        """
        Get the latest model version for a given stage.

        Returns:
            Dict with model_uri, version, run_id, metrics
        """
        versions = self.client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            logger.warning(f"No model '{model_name}' found in stage '{stage}'")
            return None

        latest = versions[0]
        run = self.client.get_run(latest.run_id)

        return {
            "model_name": model_name,
            "version": latest.version,
            "stage": latest.current_stage,
            "run_id": latest.run_id,
            "model_uri": f"models:/{model_name}/{stage}",
            "metrics": run.data.metrics,
            "params": run.data.params,
            "registered_at": latest.creation_timestamp,
        }

    def list_models(self) -> pd.DataFrame:
        """List all registered models with their versions."""
        models = self.client.search_registered_models()
        rows = []
        for m in models:
            for v in m.latest_versions:
                rows.append({
                    "model_name": m.name,
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "created_at": datetime.fromtimestamp(v.creation_timestamp / 1000).isoformat(),
                })
        return pd.DataFrame(rows)

    def compare_versions(self, model_name: str, version_a: int, version_b: int) -> dict:
        """Compare two model versions by their run metrics."""
        va = self.client.get_model_version(model_name, str(version_a))
        vb = self.client.get_model_version(model_name, str(version_b))
        run_a = self.client.get_run(va.run_id)
        run_b = self.client.get_run(vb.run_id)

        comparison = {
            "version_a": version_a,
            "version_b": version_b,
            "metrics_a": run_a.data.metrics,
            "metrics_b": run_b.data.metrics,
            "delta": {},
        }
        all_metrics = set(run_a.data.metrics.keys()) | set(run_b.data.metrics.keys())
        for m in all_metrics:
            a_val = run_a.data.metrics.get(m, 0)
            b_val = run_b.data.metrics.get(m, 0)
            comparison["delta"][m] = b_val - a_val

        return comparison

    def transition_stage(self, model_name: str, version: int, stage: str) -> dict:
        """Transition a model version to a new stage."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage=stage,
            archive_existing_versions=True,
        )
        logger.info(f"Transitioned {model_name} v{version} -> {stage}")
        return {"model_name": model_name, "version": version, "new_stage": stage}

    def archive_version(self, model_name: str, version: int) -> dict:
        """Archive a model version."""
        return self.transition_stage(model_name, version, "Archived")

    def delete_version(self, model_name: str, version: int) -> dict:
        """Delete a model version from registry."""
        self.client.delete_model_version(model_name, str(version))
        logger.info(f"Deleted {model_name} v{version}")
        return {"model_name": model_name, "version": version, "deleted": True}