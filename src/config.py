"""
MLOps Platform - Configuration
Centralized configuration using Pydantic Settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    # Platform
    app_name: str = "MLOps Platform"
    environment: str = "development"
    debug: bool = True

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///./data/mlflow.db"
    mlflow_experiment_name: str = "mlops_platform"
    mlflow_registry_model_name: str = "production_models"

    # Database (for A/B testing, metadata)
    database_url: str = "sqlite:///./data/mlops.db"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Drift detection
    drift_threshold_psi: float = 0.2
    drift_threshold_ks: float = 0.05
    drift_check_interval_minutes: int = 60

    # Retraining
    retrain_drift_threshold: float = 0.3
    retrain_performance_threshold: float = 0.05
    retrain_min_samples: int = 1000
    retrain_auto: bool = False

    # Monitoring
    prometheus_port: int = 9090
    grafana_port: int = 3000

    # A/B Testing
    ab_min_sample_size: int = 1000
    ab_significance_level: float = 0.05

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    models_dir: Path = base_dir / "data" / "models"
    reports_dir: Path = base_dir / "data" / "reports"
    artifacts_dir: Path = base_dir / "data" / "artifacts"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="MLOPS_")


settings = Settings()

# Ensure directories exist
for d in [settings.data_dir, settings.models_dir, settings.reports_dir, settings.artifacts_dir]:
    d.mkdir(parents=True, exist_ok=True)