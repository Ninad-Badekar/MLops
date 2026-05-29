import pickle

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier


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
    data_path = "data/processed/train_processed_data.csv"
    model_name = "models/model.pkl"

    n_estimators = load_params(params_path)
    train_data = load_data(data_path)
    X_train, y_train = prepare_data(train_data)
    model = train_model(X_train, y_train, n_estimators)
    save_model(model, model_name)


if __name__ == "__main__":
    main()
