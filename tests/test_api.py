import os
import pickle
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

# Add project root to path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    """Create a tiny model artifact so the API can start without a full DVC run."""
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        clf = LogisticRegression(max_iter=200, random_state=42)
        X = np.random.rand(40, 9)
        y = np.random.randint(0, 2, 40)
        clf.fit(X, y)
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
    yield


@pytest.fixture
def client(ensure_model):
    # Import after model exists so startup succeeds
    from src.main import app, prediction_log

    prediction_log.clear()
    with TestClient(app) as c:
        yield c
    prediction_log.clear()


def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["status"] == "ok"


def test_dashboard_endpoint(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Water Potability MLOps Center" in response.text
    assert "__API_KEY_JSON__" not in response.text
    assert 'const API_KEY = ""' in response.text


def test_dashboard_injects_api_key(monkeypatch, ensure_model):
    monkeypatch.setenv("API_KEY", "dashboard-secret")
    from src.main import app, prediction_log

    prediction_log.clear()
    with TestClient(app) as c:
        response = c.get("/dashboard")
        assert response.status_code == 200
        assert 'const API_KEY = "dashboard-secret"' in response.text
    prediction_log.clear()
    monkeypatch.delenv("API_KEY", raising=False)


def test_predict_endpoint_valid(client):
    payload = {
        "ph": 7.0,
        "Hardness": 200.0,
        "Solids": 20000.0,
        "Chloramines": 7.0,
        "Sulfate": 300.0,
        "Conductivity": 400.0,
        "Organic_carbon": 15.0,
        "Trihalomethanes": 60.0,
        "Turbidity": 4.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "potability" in data
    assert "latency_ms" in data
    assert data["potability"] in [0, 1]


def test_predict_endpoint_invalid(client):
    payload = {
        "Hardness": 200.0,
        "Solids": 20000.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_monitoring_stats_endpoint(client):
    response = client.get("/api/monitoring-stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "avg_latency" in data
    assert "feature_drift" in data
    assert "drift_detected" in data


def test_mlflow_info_endpoint(client):
    response = client.get("/api/mlflow-info")
    assert response.status_code == 200
    data = response.json()
    assert "configured" in data
    assert "tracking_uri" in data
    assert "ui_url" in data
    assert "metrics" in data


def test_simulate_traffic(client):
    response = client.post("/api/simulate?traffic_type=normal")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "simulated" in data["message"]

    response_stats = client.get("/api/monitoring-stats")
    assert response_stats.status_code == 200
    stats_data = response_stats.json()
    assert stats_data["total_predictions"] >= 50


def test_api_key_enforced(monkeypatch, ensure_model):
    monkeypatch.setenv("API_KEY", "test-secret")
    # Re-import not needed; Depends reads env at request time
    from src.main import app, prediction_log

    prediction_log.clear()
    with TestClient(app) as c:
        payload = {
            "ph": 7.0,
            "Hardness": 200.0,
            "Solids": 20000.0,
            "Chloramines": 7.0,
            "Sulfate": 300.0,
            "Conductivity": 400.0,
            "Organic_carbon": 15.0,
            "Trihalomethanes": 60.0,
            "Turbidity": 4.0,
        }
        denied = c.post("/predict", json=payload)
        assert denied.status_code == 401
        allowed = c.post("/predict", json=payload, headers={"X-API-Key": "test-secret"})
        assert allowed.status_code == 200
    prediction_log.clear()
    monkeypatch.delenv("API_KEY", raising=False)
