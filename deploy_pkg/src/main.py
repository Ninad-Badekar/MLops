"""Water potability prediction API with monitoring and Azure artifact loading."""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse

from src.aws_artifacts import blob_exists, download_blob_uri
from src.data_model import Water

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("water_api")

app = FastAPI(
    title="Water Potability Prediction & Monitoring API",
    description="API for predicting water potability with built-in request monitoring and drift tracking.",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "train_processed_mean.csv")
DASHBOARD_TEMPLATE_PATH = os.path.join(BASE_DIR, "src", "templates", "dashboard.html")
RING_BUFFER_SIZE = 1000

model = None
baseline_stats = {}
prediction_log: deque = deque(maxlen=RING_BUFFER_SIZE)
feature_columns = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

DEFAULT_BASELINE = {
    "ph": (7.08, 1.49),
    "Hardness": (196.36, 32.88),
    "Solids": (21835.16, 8768.57),
    "Chloramines": (7.12, 1.58),
    "Sulfate": (333.77, 36.14),
    "Conductivity": (426.21, 80.82),
    "Organic_carbon": (14.16, 3.30),
    "Trihalomethanes": (66.40, 16.17),
    "Turbidity": (3.96, 0.78),
}


def _require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """Enforce X-API-Key when API_KEY env var is set."""
    expected = os.getenv("API_KEY")
    if not expected:
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def _resolve_model_path() -> str:
    """Download model from Azure Blob when MODEL_BLOB_URI is set; otherwise use local path."""
    model_blob_uri = os.getenv("MODEL_BLOB_URI")
    if model_blob_uri:
        local_path = os.getenv("MODEL_LOCAL_PATH", MODEL_PATH)
        download_blob_uri(model_blob_uri, local_path)
        return local_path
    return MODEL_PATH


def _load_baseline_stats() -> None:
    baseline_blob_uri = os.getenv("BASELINE_BLOB_URI")
    train_path = TRAIN_DATA_PATH
    if baseline_blob_uri:
        train_path = os.path.join(BASE_DIR, "data", "processed", "train_processed_mean.csv")
        try:
            download_blob_uri(baseline_blob_uri, train_path)
        except Exception as exc:
            logger.warning("Failed to download baseline from Azure Blob: %s", exc)

    if os.path.exists(train_path):
        try:
            train_df = pd.read_csv(train_path)
            for col in feature_columns:
                if col in train_df.columns:
                    baseline_stats[col] = {
                        "mean": float(train_df[col].mean()),
                        "std": float(train_df[col].std()),
                    }
        except Exception as exc:
            logger.warning("Error loading training data for baseline stats: %s", exc)

    if not baseline_stats:
        for col in feature_columns:
            mean, std = DEFAULT_BASELINE.get(col, (0.0, 1.0))
            baseline_stats[col] = {"mean": mean, "std": std}


@app.on_event("startup")
def startup_event():
    global model

    model_path = _resolve_model_path()
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Model not found at {model_path}. "
            "Train locally, set MODEL_BLOB_URI, or place models/model.pkl before starting."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(json.dumps({"event": "model_loaded", "path": model_path}))

    _load_baseline_stats()


@app.get("/")
def index():
    return {
        "message": "Welcome to the Water Potability Prediction API!",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "dashboard": "/dashboard (GET)",
            "monitoring_stats": "/api/monitoring-stats (GET)",
            "simulate": "/api/simulate (POST)",
            "clear_logs": "/api/clear-logs (POST)",
        },
    }


@app.get("/health")
def health():
    model_ok = model is not None
    blob_ok = True
    model_blob_uri = os.getenv("MODEL_BLOB_URI")
    if model_blob_uri:
        try:
            blob_ok = blob_exists(model_blob_uri)
        except Exception:
            blob_ok = False

    status = "ok" if model_ok and blob_ok else "degraded"
    code = 200 if model_ok else 503
    payload = {
        "status": status,
        "model_loaded": model_ok,
        "blob_reachable": blob_ok,
        "model_blob_uri": model_blob_uri,
    }
    if code != 200:
        raise HTTPException(status_code=code, detail=payload)
    return payload


@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    if os.path.exists(DASHBOARD_TEMPLATE_PATH):
        with open(DASHBOARD_TEMPLATE_PATH, "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>Dashboard file not found.</h1>", status_code=404)


@app.post("/predict", dependencies=[Depends(_require_api_key)])
def predict(data: Water):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    start_time = time.perf_counter()
    feature_values = [float(data.model_dump().get(col, 0.0)) for col in feature_columns]
    sample = np.array([feature_values], dtype=float)
    prediction = int(model.predict(sample)[0])
    latency = (time.perf_counter() - start_time) * 1000

    log_prediction(data.model_dump(), prediction, latency)

    pred_str = "The water is Consumable." if prediction == 1 else "The water is Not Consumable."
    return {
        "prediction": pred_str,
        "potability": prediction,
        "latency_ms": latency,
    }


def log_prediction(features: dict, prediction: int, latency: float):
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": prediction,
        "latency": latency,
    }
    for col in feature_columns:
        entry[col] = float(features.get(col, 0.0))

    prediction_log.append(entry)
    logger.info(
        json.dumps(
            {
                "event": "prediction",
                "timestamp": entry["timestamp"],
                "prediction": prediction,
                "latency_ms": latency,
                "features": {col: entry[col] for col in feature_columns},
            }
        )
    )


@app.get("/api/monitoring-stats")
def get_monitoring_stats():
    if not prediction_log:
        return get_empty_stats()

    df = pd.DataFrame(list(prediction_log))
    total_predictions = len(df)
    avg_latency = float(df["latency"].mean())
    potable_count = int((df["prediction"] == 1).sum())
    non_potable_count = int((df["prediction"] == 0).sum())
    recent_latencies = df["latency"].tail(50).tolist()

    recent_df = df.tail(100)
    feature_drift = {}
    drift_detected = False
    drifted_features = []

    for col in feature_columns:
        if col in recent_df.columns:
            logged_mean = float(recent_df[col].mean())
            baseline_mean = baseline_stats[col]["mean"]
            baseline_std = baseline_stats[col]["std"]
            deviation = abs(logged_mean - baseline_mean)
            threshold = 1.5 * baseline_std
            drift = deviation > threshold
            if drift:
                drift_detected = True
                drifted_features.append(col)
            feature_drift[col] = {
                "baseline_mean": baseline_mean,
                "logged_mean": logged_mean,
                "deviation": deviation,
                "threshold": threshold,
                "drift": bool(drift),
            }

    recent_predictions = []
    for _, row in df.tail(10).iloc[::-1].iterrows():
        p_dict = {
            "timestamp": row["timestamp"],
            "prediction": (
                "The water is Consumable."
                if row["prediction"] == 1
                else "The water is Not Consumable."
            ),
            "latency": float(row["latency"]),
        }
        for col in feature_columns:
            p_dict[col] = float(row[col])
        recent_predictions.append(p_dict)

    return {
        "total_predictions": total_predictions,
        "avg_latency": avg_latency,
        "potable_count": potable_count,
        "non_potable_count": non_potable_count,
        "recent_latencies": recent_latencies,
        "feature_drift": feature_drift,
        "drift_detected": drift_detected,
        "drifted_features": drifted_features,
        "recent_predictions": recent_predictions,
    }


def get_empty_stats():
    feature_drift = {}
    for col in feature_columns:
        feature_drift[col] = {
            "baseline_mean": baseline_stats.get(col, {}).get("mean", 0.0),
            "logged_mean": 0.0,
            "deviation": 0.0,
            "threshold": 0.0,
            "drift": False,
        }
    return {
        "total_predictions": 0,
        "avg_latency": 0.0,
        "potable_count": 0,
        "non_potable_count": 0,
        "recent_latencies": [],
        "feature_drift": feature_drift,
        "drift_detected": False,
        "drifted_features": [],
        "recent_predictions": [],
    }


@app.post("/api/simulate", dependencies=[Depends(_require_api_key)])
def simulate_traffic(traffic_type: str = Query("normal", description="Traffic type: 'normal' or 'drifted'")):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    count = 50
    for _ in range(count):
        features = {}
        if traffic_type == "normal":
            features["ph"] = random.uniform(6.2, 7.8)
            features["Hardness"] = random.uniform(170.0, 220.0)
            features["Solids"] = random.uniform(18000.0, 25000.0)
            features["Chloramines"] = random.uniform(6.0, 8.2)
            features["Sulfate"] = random.uniform(310.0, 360.0)
            features["Conductivity"] = random.uniform(390.0, 460.0)
            features["Organic_carbon"] = random.uniform(12.0, 16.5)
            features["Trihalomethanes"] = random.uniform(55.0, 75.0)
            features["Turbidity"] = random.uniform(3.5, 4.5)
        else:
            features["ph"] = random.uniform(2.5, 4.5) if random.random() < 0.7 else random.uniform(9.5, 11.5)
            features["Hardness"] = random.uniform(196.36 - 60, 196.36 + 60)
            features["Solids"] = random.uniform(40000.0, 52000.0)
            features["Chloramines"] = random.uniform(7.12 - 3, 7.12 + 3)
            features["Sulfate"] = random.uniform(420.0, 510.0)
            features["Conductivity"] = random.uniform(426.21 - 100, 426.21 + 100)
            features["Organic_carbon"] = random.uniform(14.16 - 5, 14.16 + 5)
            features["Trihalomethanes"] = random.uniform(66.40 - 25, 66.40 + 25)
            features["Turbidity"] = random.uniform(6.5, 8.5)

        start_time = time.perf_counter()
        sample = np.array([[float(features[col]) for col in feature_columns]], dtype=float)
        prediction = int(model.predict(sample)[0])
        latency_noise = random.uniform(1.2, 5.8)
        latency = ((time.perf_counter() - start_time) * 1000) + latency_noise
        log_prediction(features, prediction, latency)

    return {"status": "success", "message": f"Successfully simulated {count} '{traffic_type}' requests."}


@app.post("/api/clear-logs", dependencies=[Depends(_require_api_key)])
def clear_logs():
    prediction_log.clear()
    return {"status": "success", "message": "Prediction logs cleared."}
