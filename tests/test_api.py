"""
tests/test_api.py - CI Automated Tests for VolleyTactix AI.
"""

import os
import sys
from fastapi.testclient import TestClient

# Ensure root & app directory are in sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TEST_DIR)
APP_DIR = os.path.join(ROOT_DIR, "app")

for p in [ROOT_DIR, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.main import app
import app.model_service as model_service

client = TestClient(app)


def test_health_endpoint():
    """Verify health endpoint returns 200 OK and valid JSON."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "system" in data
    assert data["available_models"] >= 13


def test_models_registry_endpoint():
    """Verify all 13 models and leaderboard data are returned."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "leaderboard" in data
    assert len(data["models"]) == 13

    # Check top ranked model is HistGradientBoosting
    top_model = data["models"][0]
    assert top_model["key"] == "hist_gradient_boosting"
    assert top_model["rank"] == 1


def test_static_files_exist():
    """Verify essential web application frontend files exist."""
    static_dir = os.path.join(APP_DIR, "static")
    assert os.path.isdir(static_dir), "app/static directory must exist"
    assert os.path.isfile(os.path.join(static_dir, "index.html")), "index.html must exist"
    assert os.path.isfile(os.path.join(static_dir, "style.css")), "style.css must exist"
    assert os.path.isfile(os.path.join(static_dir, "app.js")), "app.js must exist"


def test_custom_formation_prediction():
    """Verify custom formation tactical inference produces valid probability outputs."""
    # Place 6 players in a compact offensive formation
    payload = {
        "model_key": "hist_gradient_boosting",
        "ball": [4.5, 4.0],
        "players": [
            [4.5, 4.0],
            [3.0, 3.5],
            [6.0, 3.5],
            [2.5, 1.5],
            [4.5, 1.2],
            [6.5, 1.5]
        ]
    }
    response = client.post("/api/predict-custom-formation", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "prediction" in res
    assert "label" in res["prediction"]
    assert "confidence" in res["prediction"]
    assert "probabilities" in res["prediction"]
