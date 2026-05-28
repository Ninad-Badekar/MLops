import os
import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv(r"/home/neosoft/MLOps/data/processed/train_processed_data.csv")

# x_train = train_data.iloc[:, 0:-1].values

# y_train = train_data.iloc[:, -1].values
X_train = train_data.drop("Potability", axis=1).values
y_train = train_data["Potability"].values

n_estimators = yaml.safe_load(open("params.yaml"))["model_building"]["n_estimators"]
clf = RandomForestClassifier(n_estimators=n_estimators)
clf.fit(X_train, y_train)

pickle.dump(clf, open("model.pkl", "wb"))
