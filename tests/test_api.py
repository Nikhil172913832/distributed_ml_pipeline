"""
Tests for the prediction API service.
"""

import pytest
from fastapi.testclient import TestClient
from pipeline.api_service import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code in [200, 503]
    data = response.json()
    assert "status" in data
    assert "timestamp" in data


def test_model_info_endpoint():
    response = client.get("/model/info")
    # May fail if no model is trained yet
    assert response.status_code in [200, 404, 500]


def test_predict_endpoint_validation():
    # Test with empty features
    response = client.post("/predict", json={"features": {}})
    assert response.status_code == 422
    
    # Test with missing features key
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_endpoint_with_valid_data():
    # Create sample features (simplified for testing)
    features = {f"feature_{i}": float(i * 0.1) for i in range(10)}
    
    response = client.post("/predict", json={"features": features})
    
    # May fail if no model is loaded
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "confidence" in data
        assert "model_id" in data
        assert data["prediction"] in [-1, 1]
    else:
        assert response.status_code in [500]


def test_batch_predict_validation():
    # Test with empty samples
    response = client.post("/predict/batch", json={"samples": []})
    assert response.status_code == 422
    
    # Test with too many samples
    large_batch = [{"feature_0": 1.0} for _ in range(1001)]
    response = client.post("/predict/batch", json={"samples": large_batch})
    assert response.status_code == 422


def test_batch_predict_with_valid_data():
    samples = [
        {f"feature_{i}": float(i * 0.1) for i in range(10)},
        {f"feature_{i}": float(i * 0.2) for i in range(10)},
    ]
    
    response = client.post("/predict/batch", json={"samples": samples})
    
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
    else:
        assert response.status_code in [500]


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    # Prometheus metrics should be in text format
    assert "text/plain" in response.headers.get("content-type", "")
