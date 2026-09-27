from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_aoi_endpoint():
    response = client.get("/api/aoi")
    assert response.status_code == 200
    assert "name" in response.json()


def test_temporal_coverage_endpoint():
    response = client.get("/api/temporal-coverage")
    assert response.status_code == 200
    assert response.json()["start_year"] == 2022


def test_dataset_status_reports_local_download_state():
    response = client.get("/api/dataset-status")
    assert response.status_code == 200
    assert response.json()["status"] in {
        "empty", "discovered", "credentials_missing", "authenticating",
        "downloading", "downloaded", "validating", "validated", "failed",
    }
    assert "real_data_count" in response.json()
