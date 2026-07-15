"""Integration tests for the FraudShield Demo Mode Engine."""

from __future__ import annotations

from fastapi.testclient import TestClient
from src.api.main import app


def test_demo_mode_api_endpoints() -> None:
    """Test start, stop, status endpoints of Demo Mode."""
    client = TestClient(app)

    # 1. Assert demo is inactive initially
    response = client.get("/demo/status")
    assert response.status_code == 200
    status_data = response.json()
    assert status_data["is_active"] is False
    assert len(status_data["events"]) == 0

    # 2. Start the demo
    response = client.post("/demo/start")
    assert response.status_code == 200
    assert response.json() == {"status": "started"}

    # 3. Assert demo status is active
    response = client.get("/demo/status")
    assert response.status_code == 200
    status_data = response.json()
    assert status_data["is_active"] is True
    assert status_data["stage"] == 4
    assert status_data["status"] == "drift_detected"
    assert abs(status_data["accuracy"] - 80.84) < 0.05

    # 4. Check that general endpoints are overridden during demo
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_source"] == "demo_registry"

    response = client.get("/metrics")
    assert response.status_code == 200
    assert abs(response.json()["f1_score"] - 0.8084) < 0.001

    response = client.get("/drift")
    assert response.status_code == 200
    assert response.json()["drift_detected"] is True

    # 5. Stop the demo
    response = client.post("/demo/stop")
    assert response.status_code == 200
    assert response.json() == {"status": "stopped"}

    # 6. Assert demo status is inactive again
    response = client.get("/demo/status")
    assert response.status_code == 200
    assert response.json()["is_active"] is False
