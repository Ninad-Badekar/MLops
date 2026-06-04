import os
import pickle

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn


def load_params(filepath: str) -> int:
    with open(filepath) as f:
        params = yaml.safe_load(f)
    return params["model_building"]["n_estimators"]


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def prepare_data(df: pd.DataFrame):
    X = df.drop("Potability", axis=1).values
    y = df["Potability"].values
    return X, y


def train_model(X, y, n_estimators: int) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X, y)
    return clf


def save_model(model: RandomForestClassifier, filepath: str) -> None:
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    params_path = "params.yaml"
    data_path = "data/processed/train_processed_mean.csv"
    model_name = "models/model.pkl"

    n_estimators = load_params(params_path)
    train_data = load_data(data_path)
    X_train, y_train = prepare_data(train_data)

    mlflow.set_experiment("Water Potability Prediction")
    with mlflow.start_run() as run:
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("train_data_shape", str(train_data.shape))

        model = train_model(X_train, y_train, n_estimators)
        save_model(model, model_name)

        mlflow.sklearn.log_model(model, "model")

        # Save run ID to a file to be picked up by the evaluation stage
        os.makedirs("reports", exist_ok=True)
        with open("reports/mlflow_run_id.txt", "w") as f:
            f.write(run.info.run_id)


if __name__ == "__main__":
    main()
