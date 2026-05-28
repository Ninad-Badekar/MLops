import pandas as pd
import os

train_data = pd.read_csv(r"/home/neosoft/MLOps/data/raw/train_data.csv")
test_data = pd.read_csv(r"/home/neosoft/MLOps/data/raw/test_data.csv")


def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
    return df


train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

processed_data_path = os.path.join("data", "processed")
os.makedirs(processed_data_path)

train_processed_data.to_csv(
    os.path.join(processed_data_path, "train_processed_data.csv"), index=False
)
test_processed_data.to_csv(
    os.path.join(processed_data_path, "test_processed_data.csv"), index=False
)
