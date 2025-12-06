import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Setup
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Generate Synthetic Data
# We simulate 5 generic features for the existing API schema
print("Generating synthetic data...")
n_samples = 100000
n_features = 5
# 98% non-fraud (0), 2% fraud (1)
n_fraud = int(n_samples * 0.02)
n_legit = n_samples - n_fraud

# Legit transactions: N(0, 1)
X_legit = np.random.normal(loc=0, scale=1, size=(n_legit, n_features))
y_legit = np.zeros(n_legit)

# Fraud transactions: N(2, 1.5) - slightly shifted distribution
X_fraud = np.random.normal(loc=2, scale=1.5, size=(n_fraud, n_features))
y_fraud = np.ones(n_fraud)

X = np.vstack([X_legit, X_fraud])
y = np.hstack([y_legit, y_fraud])

# Shuffle
indices = np.arange(n_samples)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
print("Training Random Forest model...")
# Class_weight='balanced' helps with the imbalanced dataset
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluate
print("Evaluating model...")
y_pred = clf.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Save Model
joblib.dump(clf, MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")
