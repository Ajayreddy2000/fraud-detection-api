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

import os
import logging

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

"""
Fraud Detection API
-------------------
This FastAPI app serves a trained fraud detection model.
It exposes a /predict endpoint that accepts transaction data
and returns whether the transaction is fraudulent or not.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import time
import os
import logging
from logging.handlers import RotatingFileHandler

# -----------------------------
# 1️⃣ Logging Setup
# -----------------------------
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "api.log")
os.makedirs(LOG_DIR, exist_ok=True)  # make sure logs folder exists

# Rotating log file: prevents huge log size
handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[handler]
)

# -----------------------------
# 2️⃣ Model Loading
# -----------------------------
MODEL_PATH = "model/fraud_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    logging.info("✅ Model loaded successfully from %s", MODEL_PATH)
except Exception as e:
    logging.error("❌ Failed to load model: %s", e)
    raise RuntimeError(f"Error loading model from {MODEL_PATH}: {e}")

# -----------------------------
# 3️⃣ FastAPI App Definition
# -----------------------------
app = FastAPI(
    title="Fraud Detection API",
    description="A REST API to predict fraudulent transactions in real time.",
    version="1.0.0"
)

# -----------------------------
# 4️⃣ Request Schema
# -----------------------------
class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    transaction_type: str = Field(..., description="online or in_store")
    device: str = Field(..., description="mobile or desktop")

    # input validation
    @validator("transaction_type")
    def validate_transaction_type(cls, v):
        allowed = {"online", "in_store"}
        if v not in allowed:
            raise ValueError(f"transaction_type must be one of {allowed}")
        return v

    @validator("device")
    def validate_device(cls, v):
        allowed = {"mobile", "desktop"}
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}")
        return v


# -----------------------------
# 5️⃣ Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(transaction: Transaction):
    start_time = time.time()

    try:
        # Transform input into features (replace this with your real preprocessing)
        features = [
            [
                transaction.amount,
                1 if transaction.transaction_type == "online" else 0,
                1 if transaction.device == "mobile" else 0
            ]
        ]

        # Make prediction
        prediction = model.predict(features)[0]
        probability = float(model.predict_proba(features)[0][1])

    except Exception as e:
        logging.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    latency_ms = round((time.time() - start_time) * 1000, 2)

    # Log structured info
    logging.info(
        f"Request={transaction.dict()} | Prediction={prediction} | "
        f"Probability={probability:.3f} | Latency={latency_ms}ms"
    )

    return {
        "prediction": int(prediction),
        "fraud_probability": probability,
        "latency_ms": latency_ms
    }

# -----------------------------
# 6️⃣ Health Check Endpoint
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}

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
MODEL_PATH = "models/fraud_model.pkl"  # use your actual folder name
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
    """Simple health check endpoint"""
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

