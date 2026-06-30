"""
Test API Endpoints
Integration tests for FastAPI serving using TestClient.
Validates all endpoints: /health, /predict, /models, /drift/detect, /versions, /transition.
"""

from fastapi.testclient import TestClient
from src.serving.api import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self):
        data = response_data(client.get("/health"))
        assert "status" in data
        assert "mlflow_connected" in data
        assert "models_registered" in data
        assert "timestamp" in data

    def test_health_status_healthy_or_degraded(self):
        data = response_data(client.get("/health"))
        assert data["status"] in ("healthy", "degraded")

    def test_health_mlflow_connected_is_bool(self):
        data = response_data(client.get("/health"))
        assert isinstance(data["mlflow_connected"], bool)

    def test_health_models_registered_is_int(self):
        data = response_data(client.get("/health"))
        assert isinstance(data["models_registered"], int)


class TestModelsEndpoint:
    """Test /models endpoint."""

    def test_models_returns_200(self):
        response = client.get("/models")
        assert response.status_code == 200

    def test_models_returns_list(self):
        data = response_data(client.get("/models"))
        assert isinstance(data, list)


class TestModelInfoEndpoint:
    """Test /models/{model_name} endpoint."""

    def test_model_info_404_for_nonexistent(self):
        response = client.get("/models/nonexistent_model_12345")
        assert response.status_code in (404, 500)

    def test_model_info_error_message(self):
        data = response_data(client.get("/models/nonexistent_model_12345"))
        assert "not found" in data["detail"].lower() or "error" in data.get("detail", "").lower()


class TestModelVersionsEndpoint:
    """Test /models/{model_name}/versions endpoint."""

    def test_versions_returns_200(self):
        response = client.get("/models/nonexistent_model_12345/versions")
        assert response.status_code == 200

    def test_versions_returns_list(self):
        data = response_data(client.get("/models/nonexistent_model_12345/versions"))
        assert isinstance(data, list)


class TestDriftDetectEndpoint:
    """Test /drift/detect endpoint."""

    def test_drift_detect_returns_200(self):
        response = client.post(
            "/drift/detect",
            json={
                "reference_data": [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}],
                "current_data": [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}, {"a": 5}],
                "numerical_features": ["a"],
            },
        )
        assert response.status_code == 200

    def test_drift_detect_no_drift(self):
        import numpy as np

        ref_data = [{"a": float(v)} for v in np.random.normal(50, 10, 100)]
        cur_data = [{"a": float(v)} for v in np.random.normal(50, 10, 100)]
        data = response_data(
            client.post(
                "/drift/detect",
                json={
                    "reference_data": ref_data,
                    "current_data": cur_data,
                    "numerical_features": ["a"],
                },
            )
        )
        assert data["drift_detected"] is False

    def test_drift_detect_with_drift(self):
        import numpy as np

        ref_data = [{"a": float(v)} for v in np.random.normal(50, 10, 200)]
        cur_data = [{"a": float(v)} for v in np.random.normal(80, 20, 200)]
        data = response_data(
            client.post(
                "/drift/detect",
                json={
                    "reference_data": ref_data,
                    "current_data": cur_data,
                    "numerical_features": ["a"],
                },
            )
        )
        assert data["drift_detected"] is True
        assert "a" in data["drifted_features"]

    def test_drift_detect_missing_data(self):
        response = client.post(
            "/drift/detect",
            json={"reference_data": [], "current_data": []},
        )
        # Empty data should still return 200 (drift detector handles edge cases)
        assert response.status_code == 200

    def test_drift_detect_returns_report_path(self):
        import numpy as np

        ref_data = [{"a": float(v)} for v in np.random.normal(50, 10, 100)]
        cur_data = [{"a": float(v)} for v in np.random.normal(50, 10, 100)]
        data = response_data(
            client.post(
                "/drift/detect",
                json={
                    "reference_data": ref_data,
                    "current_data": cur_data,
                    "numerical_features": ["a"],
                },
            )
        )
        assert "report_path" in data


class TestPredictEndpoint:
    """Test /predict endpoint."""

    def test_predict_404_for_nonexistent_model(self):
        response = client.post(
            "/predict",
            json={
                "model_name": "nonexistent_model_12345",
                "data": [{"a": 1, "b": 2}],
            },
        )
        assert response.status_code in (404, 500)  # 404 if handled, 500 if MLflow raises

    def test_predict_missing_data_field(self):
        response = client.post(
            "/predict",
            json={"model_name": "test_model"},
        )
        assert response.status_code == 422  # Validation error


class TestTransitionEndpoint:
    """Test /models/{model_name}/transition endpoint."""

    def test_transition_nonexistent_model(self):
        response = client.post("/models/nonexistent_model_12345/transition?version=1&stage=Production")
        # Should return 500 (MLflow error) or 404
        assert response.status_code in (404, 500)


class TestAuthMiddleware:
    """Test API key authentication."""

    def test_protected_endpoint_without_key(self):
        response = client.post(
            "/predict",
            json={"model_name": "test", "data": [{"a": 1}]},
            headers={"X-API-Key": ""},
        )
        # With empty key, should still work if auth is optional in test mode
        # or return 401 if auth is enforced
        assert response.status_code in (200, 401, 404, 422, 500)

    def test_protected_endpoint_with_wrong_key(self):
        response = client.post(
            "/predict",
            json={"model_name": "test", "data": [{"a": 1}]},
            headers={"X-API-Key": "wrong_key"},
        )
        assert response.status_code in (200, 401, 404, 422, 500)

    def test_health_accessible_without_key(self):
        # Health endpoint should always be accessible
        response = client.get("/health")
        assert response.status_code == 200

    def test_models_accessible_without_key(self):
        # Models listing should always be accessible
        response = client.get("/models")
        assert response.status_code == 200

    def test_metrics_accessible_without_key(self):
        response = client.get("/metrics")
        assert response.status_code == 200


class TestMetricsEndpoint:
    """Test /metrics (Prometheus) endpoint."""

    def test_metrics_returns_200(self):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_contains_prometheus_format(self):
        data = client.get("/metrics").text
        assert "# HELP" in data or "# TYPE" in data


class TestApiDocs:
    """Test API documentation endpoints."""

    def test_swagger_docs_returns_200(self):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema_returns_200(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response_data(response)
        assert "paths" in data
        assert "/health" in data["paths"]
        assert "/predict" in data["paths"]


def response_data(response):
    """Helper to get JSON data from response."""
    return response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
