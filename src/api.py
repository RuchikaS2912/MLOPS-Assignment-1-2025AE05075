"""FastAPI inference service with request logging and Prometheus metrics."""

from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal
import joblib
import pandas as pd
import sklearn.compose._column_transformer as _ct
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Compatibility shim for models serialized across scikit-learn versions
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _ct._RemainderColsList = _RemainderColsList

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("heart_api")
REQUESTS = Counter("heart_api_requests_total", "Total API requests", ["method", "endpoint", "status"])
LATENCY = Histogram("heart_api_request_duration_seconds", "API request duration", ["endpoint"])

_env_path = os.getenv("MODEL_PATH")
_candidates = [
    Path(_env_path) if _env_path else None,
    Path("models/heart_pipeline.joblib"),
    Path("/Users/ruchikasreedhar/Downloads/heart_disease_mlops_colab_artifacts/models/heart_pipeline.joblib"),
]
MODEL_PATH = next((p for p in _candidates if p and p.exists()), Path("models/heart_pipeline.joblib"))
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    else:
        logger.warning("Model file not found at %s; /predict will return 503", MODEL_PATH)
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    description="Educational classifier demo; not for medical use.",
    lifespan=lifespan,
)


class PatientFeatures(BaseModel):
    age: float = Field(ge=1, le=120)
    sex: Literal[0, 1]
    cp: float = Field(ge=1, le=4)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(ge=0, le=1000)
    fbs: Literal[0, 1]
    restecg: float = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=300)
    exang: Literal[0, 1]
    oldpeak: float = Field(ge=-10, le=20)
    slope: float = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=4)
    thal: float = Field(ge=3, le=7)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    start = time.perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        elapsed = time.perf_counter() - start
        REQUESTS.labels(request.method, request.url.path, status).inc()
        LATENCY.labels(request.url.path).observe(elapsed)
        logger.info(
            "method=%s path=%s status=%s duration_seconds=%.4f", request.method, request.url.path, status, elapsed
        )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(patient: PatientFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model and provide MODEL_PATH.")
    row = pd.DataFrame([patient.model_dump()], columns=FEATURES)
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    positive_probability = float(probabilities[1])
    return {
        "prediction": prediction,
        "label": "disease_present" if prediction == 1 else "disease_not_detected_in_dataset",
        "confidence": float(max(probabilities)),
        "positive_class_probability": positive_probability,
        "disclaimer": "Educational output only; not a diagnosis.",
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
