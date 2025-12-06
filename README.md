# Fraud Detection API

A machine learning powered API for real-time fraud detection. This project demonstrates an end-to-end ML pipline including synthetic data generation, model training, and model serving.

## Features
- **Synthetic Data Generation**: Creates a realistic credit card fraud dataset.
- **Random Forest Model**: Trains a robust classifier on imbalanced data.
- **FastAPI Serving**: Exposes a `/predict` endpoint for real-time inference.
- **Logging**: Rotating logs for monitoring requests and errors.

## Quick Start

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train Model**
    Generates data and saves the model to `models/fraud_model.pkl`.
    ```bash
    python train.py
    ```

3.  **Start API**
    ```bash
    uvicorn api:app --reload
    ```

4.  **Test Prediction**
    ```bash
    python test_api.py
    ```

## API Documentation

- **Health Check**: `GET /health`
- **Predict**: `POST /predict`
    - Body:
      ```json
      {
        "feature_1": 0.5,
        "feature_2": -1.2,
        "feature_3": 3.0,
        "feature_4": 0.1,
        "feature_5": -0.5
      }
      ```
    - Response:
      ```json
      {
        "fraudulent": false,
        "fraud_probability": 0.04,
        "latency_ms": 15.2
      }
      ```
