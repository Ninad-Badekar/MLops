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
METRICS_PATH = os.path.join(BASE_DIR, "reports", "metrics.json")
MLFLOW_RUN_ID_PATH = os.path.join(BASE_DIR, "reports", "mlflow_run_id.txt")
MLFLOW_DB_PATH = os.path.join(BASE_DIR, "mlflow.db")
RING_BUFFER_SIZE = 1000

model = None
baseline_stats = {}
potable_samples: list[dict[str, float]] = []
non_potable_samples: list[dict[str, float]] = []
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


def _load_simulation_profiles(train_path: str) -> None:
    """Load labeled rows from training data for realistic traffic simulation."""
    global potable_samples, non_potable_samples
    potable_samples = []
    non_potable_samples = []

    if not os.path.exists(train_path):
        return

    try:
        train_df = pd.read_csv(train_path)
        if "Potability" not in train_df.columns:
            return

        for _, row in train_df[train_df["Potability"] == 1].iterrows():
            potable_samples.append({col: float(row[col]) for col in feature_columns if col in row})
        for _, row in train_df[train_df["Potability"] == 0].iterrows():
            non_potable_samples.append({col: float(row[col]) for col in feature_columns if col in row})
    except Exception as exc:
        logger.warning("Error loading simulation profiles: %s", exc)


def _jitter_features(features: dict[str, float], scale: float = 0.05) -> dict[str, float]:
    """Add small random noise so simulated requests are not identical."""
    jittered = {}
    for col in feature_columns:
        value = float(features.get(col, baseline_stats.get(col, {}).get("mean", 0.0)))
        jittered[col] = max(0.0, value * random.uniform(1.0 - scale, 1.0 + scale))
    return jittered


def _sample_simulation_features(traffic_type: str) -> dict[str, float]:
    """Build feature vectors from labeled training rows when available."""
    if traffic_type == "normal" and potable_samples:
        return _jitter_features(random.choice(potable_samples))
    if traffic_type != "normal" and non_potable_samples:
        return _jitter_features(random.choice(non_potable_samples), scale=0.08)

    # Fallback when training data is unavailable (e.g. minimal deploy package).
    features = {}
    if traffic_type == "normal":
        for col in feature_columns:
            mean = baseline_stats.get(col, {}).get("mean", DEFAULT_BASELINE.get(col, (0.0, 1.0))[0])
            std = baseline_stats.get(col, {}).get("std", DEFAULT_BASELINE.get(col, (0.0, 1.0))[1])
            features[col] = random.gauss(mean, std * 0.25)
    else:
        features["ph"] = random.uniform(2.5, 4.5) if random.random() < 0.7 else random.uniform(9.5, 11.5)
        features["Hardness"] = random.uniform(130.0, 260.0)
        features["Solids"] = random.uniform(40000.0, 52000.0)
        features["Chloramines"] = random.uniform(4.0, 10.0)
        features["Sulfate"] = random.uniform(420.0, 510.0)
        features["Conductivity"] = random.uniform(320.0, 560.0)
        features["Organic_carbon"] = random.uniform(9.0, 19.0)
        features["Trihalomethanes"] = random.uniform(40.0, 95.0)
        features["Turbidity"] = random.uniform(6.5, 8.5)
    return features


def _resolve_mlflow_ui_url() -> str:
    """Return MLflow UI URL from env, or a sensible local default."""
    explicit = os.getenv("MLFLOW_UI_URL", "").strip()
    if explicit:
        return explicit
    # Azure App Service sets WEBSITE_SITE_NAME; no bundled MLflow UI there.
    if os.getenv("WEBSITE_SITE_NAME"):
        return ""
    return "http://127.0.0.1:5000"


def _load_mlflow_summary() -> dict:
    """Load latest experiment summary from reports and optional local MLflow DB."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    summary: dict = {
        "configured": False,
        "tracking_uri": tracking_uri,
        "ui_url": _resolve_mlflow_ui_url(),
        "experiment_name": "Water Potability Prediction",
        "registered_model_name": "WaterPotabilityModel",
        "run_id": None,
        "metrics": {},
        "params": {},
    }

    if os.path.exists(MLFLOW_RUN_ID_PATH):
        try:
            with open(MLFLOW_RUN_ID_PATH, "r", encoding="utf-8") as f:
                run_id = f.read().strip()
                if run_id:
                    summary["run_id"] = run_id
                    summary["configured"] = True
        except OSError as exc:
            logger.warning("Failed to read MLflow run id: %s", exc)

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                summary["metrics"] = json.load(f)
                summary["configured"] = True
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read metrics.json: %s", exc)

    # Enrich from local sqlite tracking store when available (dev/CI artifact).
    if os.path.exists(MLFLOW_DB_PATH):
        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            runs = mlflow.search_runs(
                experiment_names=[summary["experiment_name"]],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if not runs.empty:
                latest = runs.iloc[0]
                summary["configured"] = True
                if summary["run_id"] is None and "run_id" in latest:
                    summary["run_id"] = str(latest["run_id"])
                for col in latest.index:
                    if col.startswith("metrics."):
                        summary["metrics"][col.replace("metrics.", "")] = float(latest[col])
                    if col.startswith("params."):
                        summary["params"][col.replace("params.", "")] = str(latest[col])
        except Exception as exc:
            logger.warning("Failed to query MLflow tracking store: %s", exc)

    return summary


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
    _load_simulation_profiles(TRAIN_DATA_PATH)


@app.get("/")
def index():
    return {
        "message": "Welcome to the Water Potability Prediction API!",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "dashboard": "/dashboard (GET)",
            "monitoring_stats": "/api/monitoring-stats (GET)",
            "mlflow_info": "/api/mlflow-info (GET)",
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
            content = f.read()
        # Inject API key so browser buttons can call protected endpoints.
        content = content.replace("__API_KEY_JSON__", json.dumps(os.getenv("API_KEY", "")))
        content = content.replace("__MLFLOW_UI_URL_JSON__", json.dumps(_resolve_mlflow_ui_url()))
        return HTMLResponse(content=content, status_code=200)
    return HTMLResponse(content="<h1>Dashboard file not found.</h1>", status_code=404)


@app.get("/api/mlflow-info")
def get_mlflow_info():
    """Expose MLflow tracking summary for the dashboard."""
    return _load_mlflow_summary()


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
        features = _sample_simulation_features(traffic_type)

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
