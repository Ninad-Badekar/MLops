import os

import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def save_data(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index=False)


def fill_missing_with_median(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if df[column].isnull().any():
            df[column] = df[column].fillna(df[column].median())
    return df


def main():
    raw_data_path = "data/raw"
    processed_data_path = "data/processed"

    train_data = load_data(os.path.join(raw_data_path, "train_data.csv"))
    test_data = load_data(os.path.join(raw_data_path, "test_data.csv"))

    train_data = fill_missing_with_median(train_data)
    test_data = fill_missing_with_median(test_data)

    os.makedirs(processed_data_path, exist_ok=True)

    save_data(train_data, os.path.join(processed_data_path, "train_processed_data.csv"))
    save_data(test_data, os.path.join(processed_data_path, "test_processed_data.csv"))


if __name__ == "__main__":
    main()
