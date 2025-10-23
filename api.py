"""
api.py — serves the trained model as a FastAPI app
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Fraud Detection API", version="1.0")

MODEL_PATH = "models/fraud_model.pkl"

class Transaction(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float

# Load model once at startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found! Please run train.py first.")
model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running 🚀"}

@app.post("/predict")
def predict(transaction: Transaction):
    data = np.array([[transaction.feature_1, transaction.feature_2, transaction.feature_3,
                      transaction.feature_4, transaction.feature_5]])
    prediction = model.predict(data)[0]
    return {"fraudulent": bool(prediction)}
