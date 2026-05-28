from re import X

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import pickle

train_data = pd.read_csv(r"/home/neosoft/MLOps/data/processed/train_processed_data.csv")

# x_train = train_data.iloc[:, 0:-1].values

# y_train = train_data.iloc[:, -1].values
X_train = train_data.drop("Potability", axis=1).values
y_train = train_data["Potability"].values
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

pickle.dump(clf, open("model.pkl", "wb"))
