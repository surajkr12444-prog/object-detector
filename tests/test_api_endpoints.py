from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_json():
    response = client.get("/", headers={"accept": "application/json"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data


def test_root_html():
    response = client.get("/", headers={"accept": "text/html"})
    assert response.status_code == 200
    assert "VisionX AI" in response.text


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "active_model" in data
    assert "available_models" in data


def test_analytics_stats():
    response = client.get("/api/analytics/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_snapshots" in data
    assert "total_detections" in data
    assert "class_distribution" in data


def test_history_snapshots_list():
    response = client.get("/api/history/snapshots")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

