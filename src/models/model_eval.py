import sys
from unittest.mock import MagicMock

# Protobuf compatibility monkeypatch for older MLflow
try:
    import google.protobuf.service
except ImportError:
    class MockService:
        RpcController = MagicMock
        RpcChannel = MagicMock
        Service = MagicMock
    sys.modules['google.protobuf.service'] = MockService
    import google.protobuf
    google.protobuf.service = MockService

import json
import os
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def load_model(filepath: str):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


def save_metrics(metrics: dict, filepath: str) -> None:
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)


def main():
    data_path = "data/processed/test_processed_mean.csv"
    model_path = "models/model.pkl"
    metrics_path = "reports/metrics.json"
    run_id_path = "reports/mlflow_run_id.txt"

    test_data = load_data(data_path)
    X_test = test_data.iloc[:, 0:-1].values
    y_test = test_data.iloc[:, -1].values

    model = load_model(model_path)
    metrics = evaluate_model(model, X_test, y_test)
    save_metrics(metrics, metrics_path)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Water Potability Prediction")
    
    run_id = None
    if os.path.exists(run_id_path):
        with open(run_id_path, "r") as f:
            run_id = f.read().strip()

    # Log metrics to MLflow, resuming the run if run_id is found
    with mlflow.start_run(run_id=run_id):
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)


if __name__ == "__main__":
    main()
