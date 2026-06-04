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

import os
import pickle

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn


def load_params(filepath: str) -> tuple[str, int]:
    with open(filepath) as f:
        params = yaml.safe_load(f)
    model_type = params["model_building"].get("model_type", "random_forest")
    n_estimators = params["model_building"].get("n_estimators", 100)
    return model_type, n_estimators


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def prepare_data(df: pd.DataFrame):
    X = df.drop("Potability", axis=1).values
    y = df["Potability"].values
    return X, y


def train_model(X, y, model_type: str, n_estimators: int):
    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif model_type == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    elif model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    clf.fit(X, y)
    return clf


def save_model(model, filepath: str) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    params_path = "params.yaml"
    data_path = "data/processed/train_processed_mean.csv"
    model_name = "models/model.pkl"

    model_type, n_estimators = load_params(params_path)
    train_data = load_data(data_path)
    X_train, y_train = prepare_data(train_data)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Water Potability Prediction")
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("train_data_shape", str(train_data.shape))

        model = train_model(X_train, y_train, model_type, n_estimators)
        save_model(model, model_name)

        # Log the model and register it in the Model Registry
        mlflow.sklearn.log_model(
            model, 
            "model", 
            registered_model_name="WaterPotabilityModel"
        )

        # Save run ID to a file to be picked up by the evaluation stage
        os.makedirs("reports", exist_ok=True)
        with open("reports/mlflow_run_id.txt", "w") as f:
            f.write(run.info.run_id)


if __name__ == "__main__":
    main()
