import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_params(filepath: str) -> float:
    with open(filepath) as f:
        params = yaml.safe_load(f)
    return params["data_collection"]["test_size"]


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def split_data(data: pd.DataFrame, test_size: float):
    return train_test_split(data, test_size=test_size, random_state=42)


def save_data(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index=False)


def main():
    data_filepath = "data/external/water_potability.csv"
    params_filepath = "params.yaml"
    raw_data_path = os.path.join("data", "raw")

    data = load_data(data_filepath)
    test_size = load_params(params_filepath)
    train_data, test_data = split_data(data, test_size)

    os.makedirs(raw_data_path, exist_ok=True)

    save_data(train_data, os.path.join(raw_data_path, "train_data.csv"))
    save_data(test_data, os.path.join(raw_data_path, "test_data.csv"))


if __name__ == "__main__":
    main()
