import os
import time
import pickle
import random
import csv
from datetime import datetime
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from src.data_model import Water

app = FastAPI(
    title="Water Potability Prediction & Monitoring API",
    description="API for predicting water potability with built-in request monitoring and drift tracking.",
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
MONITORING_DIR = os.path.join(BASE_DIR, "data", "monitoring")
LOG_PATH = os.path.join(MONITORING_DIR, "predictions_log.csv")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "train_processed_mean.csv")
DASHBOARD_TEMPLATE_PATH = os.path.join(BASE_DIR, "src", "templates", "dashboard.html")

# Global variables
model = None
baseline_stats = {}
feature_columns = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]

@app.on_event("startup")
def startup_event():
    global model
    
    # 1. Load model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        # Fallback dummy model if file is not found (for safety)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10)
        X_dummy = np.random.rand(10, 9)
        y_dummy = np.random.randint(0, 2, 10)
        model.fit(X_dummy, y_dummy)

    # 2. Setup logs file
    os.makedirs(MONITORING_DIR, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + feature_columns + ["prediction", "latency"])

    # 3. Load baseline training stats for drift calculation
    if os.path.exists(TRAIN_DATA_PATH):
        try:
            train_df = pd.read_csv(TRAIN_DATA_PATH)
            for col in feature_columns:
                if col in train_df.columns:
                    baseline_stats[col] = {
                        "mean": float(train_df[col].mean()),
                        "std": float(train_df[col].std())
                    }
        except Exception as e:
            print(f"Error loading training data for baseline stats: {e}")
            
    # Default baseline stats if training data is missing
    if not baseline_stats:
        # Values matching typical distributions in water_potability dataset
        defaults = {
            "ph": (7.08, 1.49),
            "Hardness": (196.36, 32.88),
            "Solids": (21835.16, 8768.57),
            "Chloramines": (7.12, 1.58),
            "Sulfate": (333.77, 36.14),
            "Conductivity": (426.21, 80.82),
            "Organic_carbon": (14.16, 3.30),
            "Trihalomethanes": (66.40, 16.17),
            "Turbidity": (3.96, 0.78)
        }
        for col in feature_columns:
            mean, std = defaults.get(col, (0.0, 1.0))
            baseline_stats[col] = {"mean": mean, "std": std}


@app.get("/")
def index():
    return {
        "message": "Welcome to the Water Potability Prediction API!",
        "endpoints": {
            "predict": "/predict (POST)",
            "dashboard": "/dashboard (GET)",
            "monitoring_stats": "/api/monitoring-stats (GET)",
            "simulate": "/api/simulate (POST)",
            "clear_logs": "/api/clear-logs (POST)"
        }
    }


@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    if os.path.exists(DASHBOARD_TEMPLATE_PATH):
        with open(DASHBOARD_TEMPLATE_PATH, "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>Dashboard file not found.</h1>", status_code=404)


@app.post("/predict")
def predict(data: Water):
    start_time = time.perf_counter()
    
    # Run inference
    sample = pd.DataFrame([data.model_dump()])
    prediction = int(model.predict(sample)[0])
    
    latency = (time.perf_counter() - start_time) * 1000 # in ms
    
    # Log prediction details to CSV
    log_prediction(data.model_dump(), prediction, latency)
    
    pred_str = "The water is Consumable." if prediction == 1 else "The water is Not Consumable."
    return {
        "prediction": pred_str,
        "potability": prediction,
        "latency_ms": latency
    }


def log_prediction(features: dict, prediction: int, latency: float):
    row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    for col in feature_columns:
        row.append(features.get(col, 0.0))
    row.append(prediction)
    row.append(latency)
    
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


@app.get("/api/monitoring-stats")
def get_monitoring_stats():
    if not os.path.exists(LOG_PATH):
        return get_empty_stats()
        
    try:
        df = pd.read_csv(LOG_PATH)
    except Exception:
        return get_empty_stats()
        
    if df.empty or len(df) == 0:
        return get_empty_stats()
        
    total_predictions = len(df)
    avg_latency = float(df["latency"].mean())
    potable_count = int((df["prediction"] == 1).sum())
    non_potable_count = int((df["prediction"] == 0).sum())
    
    recent_latencies = df["latency"].tail(50).tolist()
    
    # Calculate drift metrics (on last 100 predictions)
    recent_df = df.tail(100)
    feature_drift = {}
    drift_detected = False
    drifted_features = []
    
    for col in feature_columns:
        if col in recent_df.columns:
            logged_mean = float(recent_df[col].mean())
            baseline_mean = baseline_stats[col]["mean"]
            baseline_std = baseline_stats[col]["std"]
            
            # Check if logged mean deviates by more than 1.5 standard deviation
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
                "drift": bool(drift)
            }
            
    # Recent logs for table display
    recent_predictions = []
    for idx, row in df.tail(10).iterrows():
        p_dict = {
            "timestamp": row["timestamp"],
            "prediction": "The water is Consumable." if row["prediction"] == 1 else "The water is Not Consumable.",
            "latency": float(row["latency"])
        }
        for col in feature_columns:
            p_dict[col] = float(row[col])
        recent_predictions.append(p_dict)
        
    # Reverse order for table display
    recent_predictions.reverse()

    return {
        "total_predictions": total_predictions,
        "avg_latency": avg_latency,
        "potable_count": potable_count,
        "non_potable_count": non_potable_count,
        "recent_latencies": recent_latencies,
        "feature_drift": feature_drift,
        "drift_detected": drift_detected,
        "drifted_features": drifted_features,
        "recent_predictions": recent_predictions
    }


def get_empty_stats():
    feature_drift = {}
    for col in feature_columns:
        feature_drift[col] = {
            "baseline_mean": baseline_stats.get(col, {}).get("mean", 0.0),
            "logged_mean": 0.0,
            "deviation": 0.0,
            "threshold": 0.0,
            "drift": False
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
        "recent_predictions": []
    }


@app.post("/api/simulate")
def simulate_traffic(traffic_type: str = Query("normal", description="Traffic type: 'normal' or 'drifted'")):
    if model is None:
        return {"status": "error", "message": "Model not loaded."}
        
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
            # Inject extreme values to trigger drift on ph, solids, sulfate, and turbidity
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
        
        # Format df for prediction
        sample = pd.DataFrame([features])
        prediction = int(model.predict(sample)[0])
        
        # Introduce mock processing latency variation
        latency_noise = random.uniform(1.2, 5.8)
        latency = ((time.perf_counter() - start_time) * 1000) + latency_noise
        
        log_prediction(features, prediction, latency)
        
    return {"status": "success", "message": f"Successfully simulated {count} '{traffic_type}' requests."}


@app.post("/api/clear-logs")
def clear_logs():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + feature_columns + ["prediction", "latency"])
    return {"status": "success", "message": "Prediction logs cleared."}
