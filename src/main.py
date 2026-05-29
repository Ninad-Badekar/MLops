import os
import pickle

import pandas as pd
from fastapi import FastAPI
from data_model import Water

app = FastAPI(
    title="Water Potability Prediction API",
    description="API for predicting water potability based on various features.",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@app.get("/")
def index():
    return "Welcome to the Water Potability Prediction API!"


@app.post("/predict")
def predict(data: Water):
    sample = pd.DataFrame([data.model_dump()])
    prediction = model.predict(sample)

    if prediction[0] == 1:
        return {"prediction": "The water is Consumable."}
    return {"prediction": "The water is Not Consumable."}
