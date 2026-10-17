from fastapi.testclient import TestClient
from src.api import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_prediction_rejects_invalid_input():
    with TestClient(app) as client:
        response = client.post("/predict", json={"age": 999})
        assert response.status_code == 422


def test_predict_returns_prediction_and_confidence(monkeypatch):
    import numpy as np
    import src.api as api

    class DummyModel:
        def predict(self, X):
            return np.array([1])

        def predict_proba(self, X):
            return np.array([[0.2, 0.8]])

    monkeypatch.setattr(api, "model", DummyModel())
    payload = {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0,
               "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 2, "ca": 0, "thal": 3}
    client = TestClient(api.app)
    body = client.post("/predict", json=payload).json()
    assert body["prediction"] == 1
    assert body["confidence"] == 0.8
