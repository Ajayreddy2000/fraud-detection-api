"""
api.py — Fraud Detection API
----------------------------
Serves a trained fraud detection model via FastAPI.
Logs predictions, latency, and errors in rotating log files.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import time
import os
import logging
from logging.handlers import RotatingFileHandler

# -----------------------------
# 1️⃣ Logging Setup
# -----------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "api.log")

handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[handler],
)

# -----------------------------
# 2️⃣ Model Loading
# -----------------------------
MODEL_PATH = "models/fraud_model.pkl"  # match your training output folder
if not os.path.exists(MODEL_PATH):
    logging.error("❌ Model not found at %s. Run train.py first.", MODEL_PATH)
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")

try:
    model = joblib.load(MODEL_PATH)
    logging.info("✅ Model loaded successfully from %s", MODEL_PATH)
except Exception as e:
    logging.error("❌ Error loading model: %s", e)
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

# -----------------------------
# 3️⃣ FastAPI App
# -----------------------------
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection model served via FastAPI",
    version="1.0.0",
)

# -----------------------------
# 4️⃣ Request Schema
# -----------------------------
class Transaction(BaseModel):
    feature_1: float = Field(..., description="Numeric input feature 1")
    feature_2: float = Field(..., description="Numeric input feature 2")
    feature_3: float = Field(..., description="Numeric input feature 3")
    feature_4: float = Field(..., description="Numeric input feature 4")
    feature_5: float = Field(..., description="Numeric input feature 5")

# -----------------------------
# 5️⃣ Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "Fraud Detection API is running 🚀"}

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict(transaction: Transaction):
    """Make a fraud prediction for a single transaction"""
    start_time = time.time()

    try:
        # Convert input into numpy array
        X = np.array([[
            transaction.feature_1,
            transaction.feature_2,
            transaction.feature_3,
            transaction.feature_4,
            transaction.feature_5,
        ]])

        # Predict
        prediction = model.predict(X)[0]
        probability = float(model.predict_proba(X)[0][1])
    except Exception as e:
        logging.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    latency_ms = round((time.time() - start_time) * 1000, 2)
    logging.info(
        f"Request={transaction.dict()} | Prediction={prediction} | "
        f"Prob={probability:.3f} | Latency={latency_ms}ms"
    )

    return {
        "fraudulent": bool(prediction),
        "fraud_probability": probability,
        "latency_ms": latency_ms,
    }
