import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add project root to path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.main import app, LOG_PATH

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data


def test_dashboard_endpoint(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Water Potability MLOps Center" in response.text


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
        "Turbidity": 4.0
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
        "Solids": 20000.0
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


def test_simulate_traffic(client):
    response = client.post("/api/simulate?traffic_type=normal")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "simulated" in data["message"]
    
    assert os.path.exists(LOG_PATH)
    
    response_stats = client.get("/api/monitoring-stats")
    assert response_stats.status_code == 200
    stats_data = response_stats.json()
    assert stats_data["total_predictions"] >= 50


if __name__ == "__main__":
    print("Running API unit tests...")
    with TestClient(app) as c:
        print("Executing test_read_root...")
        test_read_root(c)
        print("- test_read_root passed")
        
        print("Executing test_dashboard_endpoint...")
        test_dashboard_endpoint(c)
        print("- test_dashboard_endpoint passed")
        
        print("Executing test_predict_endpoint_valid...")
        test_predict_endpoint_valid(c)
        print("- test_predict_endpoint_valid passed")
        
        print("Executing test_predict_endpoint_invalid...")
        test_predict_endpoint_invalid(c)
        print("- test_predict_endpoint_invalid passed")
        
        print("Executing test_monitoring_stats_endpoint...")
        test_monitoring_stats_endpoint(c)
        print("- test_monitoring_stats_endpoint passed")
        
        print("Executing test_simulate_traffic...")
        test_simulate_traffic(c)
        print("- test_simulate_traffic passed")
        
        print("\nALL API TESTS PASSED SUCCESSFULLY!")
