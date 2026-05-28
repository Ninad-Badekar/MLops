import numpy as np
import pandas as pd
import os
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

test_data = pd.read_csv(r"/home/neosoft/MLOps/data/processed/test_processed_data.csv")
x_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:, -1].values

model = pickle.load(open("model.pkl", "rb"))

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
