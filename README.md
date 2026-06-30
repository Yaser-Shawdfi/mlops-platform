# MLOps Platform - Enterprise ML Lifecycle Management

A production-grade MLOps platform that manages the full machine learning lifecycle: experiment tracking, model registry, drift detection, automated retraining, A/B testing, and monitoring. Designed to unify and operationalize existing ML models (readmission predictor, credit scoring, cancer detection) under a single management layer.

## Architecture

```
                    +-------------------+
                    |   Data Sources     |
                    |  (Reference + Live)|
                    +---------+---------+
                              |
                    +---------v---------+
                    |  Drift Detector     |
                    |  (PSI / KS Test)   |
                    +---------+---------+
                              |
                    +---------v---------+
                    |  Retrain Trigger   |
                    |  (Decision Engine) |
                    +---------+---------+
                              |
              +---------------+---------------+
              |                               |
    +---------v---------+           +---------v---------+
    |  Retrain Pipeline |           |  Alert Manager    |
    |  (Prefect)        |           |  (Slack/Email)    |
    +---------+---------+           +-------------------+
              |
    +---------v---------+
    |  MLflow Registry   |
    |  (Version + Stage) |
    +---------+---------+
              |
    +---------v---------+
    |  FastAPI Serving   |
    |  (/predict)       |
    +---------+---------+
              |
    +---------v---------+
    |  Prometheus +     |
    |  Grafana Dashboard|
    +-------------------+
              |
    +---------v---------+
    |  A/B Test Manager  |
    |  (Z-test, KS)     |
    +-------------------+
```

## Project Structure

```
mlops-platform/
|-- src/
|   |-- config.py                    # Centralized configuration (Pydantic Settings)
|   |-- registry/
|   |   |-- model_registry.py        # MLflow model registry: log, register, version, stage
|   |-- drift/
|   |   |-- drift_detector.py        # PSI, KS test, chi-square drift detection
|   |-- pipeline/
|   |   |-- retrain_pipeline.py      # Automated retraining with trigger logic
|   |-- serving/
|   |   |-- api.py                    # FastAPI: /predict, /drift/detect, /models, /health
|   |-- monitoring/
|   |   |-- metrics_collector.py     # Prometheus metrics + AlertManager
|   |-- ab_testing/
|   |   |-- ab_test_manager.py       # A/B test creation, routing, statistical evaluation
|   |-- models/
|       |-- model_wrappers.py        # Synthetic data generators + training functions
|-- tests/
|   |-- test_drift.py                # PSI, feature/target/prediction drift tests
|   |-- test_models.py               # Data generation + model training tests
|   |-- test_ab_testing.py           # A/B test lifecycle tests
|   |-- test_monitoring.py           # Metrics + alerting tests
|   |-- test_pipeline.py             # Retraining trigger + execution tests
|-- scripts/
|   |-- demo.py                      # Full lifecycle demo script
|-- docker/
|   |-- Dockerfile.mlflow            # MLflow tracking server
|   |-- Dockerfile.api               # MLOps API server
|-- monitoring/
|   |-- prometheus.yml               # Prometheus scrape config
|   |-- prometheus_rules.yml         # Alert rules (drift, latency, degradation)
|   |-- grafana/
|       |-- provisioning/            # Auto-provisioned datasources + dashboards
|       |-- dashboards/
|           |-- mlops_dashboard.json # Pre-built Grafana dashboard
|-- .github/workflows/ci.yml         # CI/CD: test, lint, docker build
|-- docker-compose.yml               # Full stack: MLflow + API + Prometheus + Grafana
|-- requirements.txt
|-- .gitignore
```

## Key Features

| Feature | Implementation | Module |
|---|---|---|
| **Experiment Tracking** | MLflow tracking server with run logging, params, metrics, artifacts | `src/registry/model_registry.py` |
| **Model Registry** | Versioned model registration with stage transitions (Staging/Production/Archived) | `src/registry/model_registry.py` |
| **Drift Detection** | Population Stability Index (PSI), KS test, chi-square for feature/target/prediction drift | `src/drift/drift_detector.py` |
| **Automated Retraining** | Trigger-based pipeline with configurable thresholds, audit trail, auto-deploy | `src/pipeline/retrain_pipeline.py` |
| **Model Serving** | FastAPI with /predict, /models, /drift/detect, /health, /metrics endpoints | `src/serving/api.py` |
| **Monitoring** | Prometheus metrics (predictions, latency, drift, retrain) + Grafana dashboard | `src/monitoring/metrics_collector.py` |
| **Alerting** | Alert manager with severity levels, persistence, auto-evaluation | `src/monitoring/metrics_collector.py` |
| **A/B Testing** | Traffic routing, outcome recording, two-proportion z-test, effect size (Cohen's h) | `src/ab_testing/ab_test_manager.py` |
| **CI/CD** | GitHub Actions: pytest, ruff lint, ruff format, Docker build | `.github/workflows/ci.yml` |
| **Docker Compose** | Full stack: MLflow + API + Prometheus + Grafana with health checks | `docker-compose.yml` |

## Quick Start

### Option 1: Run Demo (No Docker Required)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full lifecycle demo
python scripts/demo.py
```

The demo covers: data generation -> model training -> MLflow logging -> drift detection -> alerting -> retraining pipeline -> A/B test evaluation.

### Option 2: Run Full Stack with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Services available at:
# - API:         http://localhost:8000/docs
# - MLflow:      http://localhost:5000
# - Prometheus:  http://localhost:9090
# - Grafana:     http://localhost:3000 (admin/admin)
```

### Option 3: Run Tests

```bash
# Install dependencies
pip install -r requirements.txt pytest pytest-cov

# Run all tests
python -m pytest tests/ -v --tb=short

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

## API Endpoints

| Method | Endpoint | Auth | Rate Limit | Description |
|---|---|---|---|---|
| GET | `/health` | Public | 100/min | Platform health check |
| POST | `/predict` | API Key | 30/min | Serve predictions from a registered model |
| GET | `/models` | Public | 100/min | List all registered models |
| GET | `/models/{name}` | Public | 100/min | Get info about a specific model |
| GET | `/models/{name}/versions` | Public | 100/min | List all versions of a model |
| POST | `/models/{name}/transition` | API Key | 100/min | Transition model to a new stage |
| POST | `/drift/detect` | API Key | 100/min | Detect drift between datasets |
| GET | `/metrics` | Public | 100/min | Prometheus metrics endpoint |
| GET | `/docs` | Public | - | Swagger UI |

### Authentication

Set `MLOPS_API_KEY` environment variable to enable API key authentication. When set, protected endpoints require an `X-API-Key` header.

```bash
# Enable auth
export MLOPS_API_KEY="your-secret-key"

# Call protected endpoint
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "readmission_predictor", "data": [{"age": 65}]}'
```

When `MLOPS_API_KEY` is empty (default), auth is disabled (development mode).

### API Examples

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Get model info
curl http://localhost:8000/models/readmission_predictor

# Get model versions
curl http://localhost:8000/models/readmission_predictor/versions

# Predict (requires registered model)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "readmission_predictor",
    "data": [{"age": 65, "length_of_stay": 5, "num_diagnoses": 3}],
    "model_stage": "Production"
  }'

# Detect drift
curl -X POST http://localhost:8000/drift/detect \
  -H "Content-Type: application/json" \
  -d '{
    "reference_data": [{"age": 50}, {"age": 55}, {"age": 45}, {"age": 60}],
    "current_data": [{"age": 70}, {"age": 75}, {"age": 65}, {"age": 80}],
    "numerical_features": ["age"]
  }'

# Transition model stage
curl -X POST "http://localhost:8000/models/readmission_predictor/transition?version=2&stage=Production"

# Prometheus metrics
curl http://localhost:8000/metrics
```

## Drift Detection

The platform uses multiple statistical methods:

| Method | Use Case | Threshold |
|---|---|---|
| **PSI (Population Stability Index)** | Numerical feature drift | PSI > 0.2 = significant drift |
| **KS Test** | Numerical target/prediction drift | p-value < 0.05 = drift |
| **Chi-Square** | Categorical target drift | p-value < 0.05 = drift |

PSI Interpretation:
- PSI < 0.1: No significant change
- PSI 0.1 - 0.2: Moderate drift, monitor
- PSI > 0.2: Significant drift, retrain

## Retraining Pipeline

Retraining is triggered when:
1. **Feature drift**: 3+ features exceed PSI threshold
2. **Performance degradation**: Any metric drops by more than `retrain_performance_threshold` (default: 0.05)

Pipeline steps:
1. Drift detection (feature + target + prediction)
2. Trigger evaluation
3. Retrain on combined reference + current data
4. Register new version to MLflow (Staging)
5. Optional auto-deploy to Production

## A/B Testing Framework

| Step | Action |
|---|---|
| Create | Define test with model versions, traffic split, primary metric |
| Route | Traffic routing based on configurable split (default 50/50) |
| Record | Log predictions + ground truth outcomes |
| Evaluate | Two-proportion z-test with significance level (default 0.05) |
| Recommend | deploy_b / keep_a / inconclusive based on p-value |

## Monitoring Dashboard

The pre-built Grafana dashboard (`monitoring/grafana/dashboards/mlops_dashboard.json`) includes:

- Predictions served (total counter)
- Model AUC and Accuracy gauges
- Drift alert counter
- Prediction latency (P50/P95/P99 histograms)
- Drift PSI by feature
- Predictions over time (rate graph)
- Retraining pipeline status (completed/failed/skipped)

## Prometheus Alert Rules

| Alert | Condition | Severity |
|---|---|---|
| ModelDriftDetected | `mlops_drift_detected_total > 0` | Warning |
| ModelPerformanceDegraded | `mlops_model_auc < 0.7` | Critical |
| PredictionLatencyHigh | P95 latency > 1s for 2m | Warning |
| RetrainPipelineFailures | 3+ failures in 1h | Critical |

## Configuration

All settings are in `src/config.py` with environment variable overrides (prefix `MLOPS_`):

| Setting | Default | Description |
|---|---|---|
| `MLOPS_MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `MLOPS_DRIFT_THRESHOLD_PSI` | `0.2` | PSI threshold for drift |
| `MLOPS_DRIFT_THRESHOLD_KS` | `0.05` | P-value threshold for KS/chi-square |
| `MLOPS_RETRAIN_DRIFT_THRESHOLD` | `0.3` | PSI to trigger retrain |
| `MLOPS_RETRAIN_PERFORMANCE_THRESHOLD` | `0.05` | Metric drop to trigger retrain |
| `MLOPS_RETRAIN_MIN_SAMPLES` | `1000` | Minimum samples for retrain |
| `MLOPS_RETRAIN_AUTO` | `False` | Auto-deploy retrained model |
| `MLOPS_AB_MIN_SAMPLE_SIZE` | `1000` | Min samples per A/B variant |
| `MLOPS_AB_SIGNIFICANCE_LEVEL` | `0.05` | P-value threshold for A/B test |

## Integration with Existing Projects

The platform is designed to wrap existing models. Current model wrappers:

| Model | Source Project | Data Generator | Type |
|---|---|---|---|
| `readmission_predictor` | readmission-predictor | Synthetic patient data (5K samples) | XGBoost |
| `credit_scorer` | credit-scoring-xai | Synthetic credit data (5K samples) | XGBoost |

To add a new model:
1. Add a data generator function in `src/models/model_wrappers.py`
2. Add a training function returning `(model, metrics, params)`
3. Register it in `MODEL_REGISTRY` dict
4. The platform handles logging, registration, drift detection, and serving automatically

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Model Registry | MLflow 2.22 |
| Drift Detection | Custom PSI + SciPy (KS, chi-square) |
| Pipeline Orchestration | Prefect 2.20 |
| Model Serving | FastAPI + Uvicorn |
| Monitoring | Prometheus + Grafana |
| Models | XGBoost, LightGBM, scikit-learn |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |
| Testing | pytest + pytest-cov |

## License

MIT