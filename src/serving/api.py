"""
FastAPI Model Serving
Serves models from MLflow registry with Prometheus metrics,
health checks, and model info endpoints.
"""

from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import pandas as pd
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from src.config import settings
from src.registry.model_registry import ModelRegistry
from src.drift.drift_detector import DriftDetector


# --- Rate Limiting ---

limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])


# --- API Key Authentication ---

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Endpoints that don't require authentication
PUBLIC_ENDPOINTS = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key for protected endpoints."""
    # If no API key is configured in settings, auth is disabled
    configured_key = os.environ.get("MLOPS_API_KEY", "")

    if not configured_key:
        # Auth disabled (development mode)
        return "anonymous"

    if api_key == configured_key:
        return api_key

    raise HTTPException(status_code=401, detail="Invalid or missing API key. Provide X-API-Key header.")


# --- Request/Response Models ---


class PredictionRequest(BaseModel):
    model_name: str = Field(..., description="Registered model name")
    data: List[Dict[str, Any]] = Field(..., description="Input data as list of dicts")
    model_version: Optional[str] = Field(None, description="Specific version, or latest if None")
    model_stage: Optional[str] = Field("Production", description="Model stage: Production, Staging")


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    predictions: List[Any]
    prediction_count: int
    served_at: str


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    stage: str
    run_id: str
    metrics: Dict[str, float]
    params: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    mlflow_connected: bool
    models_registered: int
    timestamp: str


class DriftRequest(BaseModel):
    reference_data: List[Dict[str, Any]]
    current_data: List[Dict[str, Any]]
    numerical_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None


# --- FastAPI App ---

app = FastAPI(
    title="MLOps Platform API",
    description="Enterprise ML lifecycle management: model serving, drift detection, retraining",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )


# Prometheus metrics
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
).instrument(app).expose(app, endpoint="/metrics")


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check platform health."""
    try:
        registry = ModelRegistry()
        models_df = registry.list_models()
        mlflow_ok = True
        model_count = len(models_df)
    except Exception:
        mlflow_ok = False
        model_count = 0

    return HealthResponse(
        status="healthy" if mlflow_ok else "degraded",
        mlflow_connected=mlflow_ok,
        models_registered=model_count,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("30/minute")
async def predict(request: Request, prediction_req: PredictionRequest, api_key: str = Depends(verify_api_key)):
    """Serve predictions from a registered model."""
    try:
        registry = ModelRegistry()
        model_info = registry.get_latest_model(prediction_req.model_name, stage=prediction_req.model_stage)
        if model_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{prediction_req.model_name}' not found in stage '{prediction_req.model_stage}'",
            )

        # Load model from MLflow
        import mlflow

        model_uri = model_info["model_uri"]
        model = mlflow.pyfunc.load_model(model_uri)

        # Convert input
        df = pd.DataFrame(prediction_req.data)
        predictions = model.predict(df).tolist()

        # Log prediction stats for drift tracking
        logger.info(f"Served {len(predictions)} predictions from {prediction_req.model_name} v{model_info['version']}")

        return PredictionResponse(
            model_name=prediction_req.model_name,
            model_version=model_info["version"],
            predictions=predictions,
            prediction_count=len(predictions),
            served_at=datetime.now().isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=List[Dict])
async def list_models():
    """List all registered models."""
    registry = ModelRegistry()
    df = registry.list_models()
    return df.to_dict(orient="records")


@app.get("/models/{model_name}", response_model=ModelInfoResponse)
async def get_model_info(model_name: str, stage: str = "Production"):
    """Get info about a specific model."""
    try:
        registry = ModelRegistry()
        info = registry.get_latest_model(model_name, stage=stage)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        return ModelInfoResponse(
            model_name=info["model_name"],
            version=info["version"],
            stage=info["stage"],
            run_id=info["run_id"],
            metrics=info["metrics"],
            params=info["params"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found: {e}")


@app.post("/drift/detect", response_model=Dict)
async def detect_drift(request: DriftRequest, api_key: str = Depends(verify_api_key)):
    """Detect data drift between reference and current data."""
    detector = DriftDetector()
    ref_df = pd.DataFrame(request.reference_data)
    cur_df = pd.DataFrame(request.current_data)
    result = detector.detect_feature_drift(
        ref_df,
        cur_df,
        numerical_features=request.numerical_features,
        categorical_features=request.categorical_features,
    )
    return result


@app.get("/models/{model_name}/versions")
async def get_versions(model_name: str):
    """List all versions of a model."""
    registry = ModelRegistry()
    versions = registry.client.search_model_versions(f"name='{model_name}'")
    return [
        {
            "version": v.version,
            "stage": v.current_stage,
            "run_id": v.run_id,
            "created_at": datetime.fromtimestamp(v.creation_timestamp / 1000).isoformat(),
        }
        for v in versions
    ]


@app.post("/models/{model_name}/transition")
async def transition_model(model_name: str, version: int, stage: str, api_key: str = Depends(verify_api_key)):
    """Transition a model version to a new stage."""
    try:
        registry = ModelRegistry()
        result = registry.transition_stage(model_name, version, stage)
        return result
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to transition {model_name} v{version}: {e}")


@app.on_event("startup")
async def startup_event():
    logger.info(f"MLOps Platform API starting on {settings.api_host}:{settings.api_port}")
    logger.info(f"MLflow tracking URI: {settings.mlflow_tracking_uri}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("MLOps Platform API shutting down")
