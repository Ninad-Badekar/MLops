import os
from fastapi import FastAPI
import pickle
import pandas as pd
from data_model import Water

app = FastAPI(
    title="Water Potability Prediction API",
    description="API for predicting water potability based on various features.",
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@app.get("/")
def index():
    return "Welcome to the Water Potability Prediction API!"


@app.post("/predict")
def predict(data: Water):
    sample = pd.DataFrame(
        [
            {
                "ph": data.ph,
                "Hardness": data.Hardness,
                "Solids": data.Solids,
                "Chloramines": data.Chloramines,
                "Sulfate": data.Sulfate,
                "Conductivity": data.Conductivity,
                "Organic_carbon": data.Organic_carbon,
                "Trihalomethanes": data.Trihalomethanes,
                "Turbidity": data.Turbidity,
            }
        ]
    )

    prediction = model.predict(sample)

    if prediction[0] == 1:
        return {"prediction": "The water is Consumable."}
    else:
        return {"prediction": "The water is Not Consumable."}
