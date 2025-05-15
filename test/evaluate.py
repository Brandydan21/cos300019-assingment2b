from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import sys

# ---------------- Import your processed data ----------------
path = os.path.abspath("../data_processing")
sys.path.append(path)
import data_preprocessing

X_test = data_preprocessing.X_test
y_test = data_preprocessing.y_test

# ---------------- Load scaler (use any known SCAT key) ----------------
scalers = joblib.load("../processed_data/scalers_by_scat.joblib")
scaler = scalers[list(scalers.keys())[0]]  # use first available

# Ensure y_test has shape (n, 1) for inverse transform
if y_test.ndim == 1:
    y_test = y_test.reshape(-1, 1)

y_true = scaler.inverse_transform(y_test)

# ---------------- Load models ----------------
models = {
    "LSTM": load_model("../training/models/lstm_model.keras"),
    "GRU": load_model("../training/models/gru_model.keras"),
    "Dense NN": load_model("../training/models/dense_model.keras")
}

# ---------------- Evaluate ----------------
for name, model in models.items():
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n{name} Evaluation:")
    print(f"  MSE: {mse:.2f}")
    print(f"  MAE: {mae:.2f}")

    # Plot first 100 points
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:100], label="Actual", marker='o')
    plt.plot(y_pred[:100], label="Predicted", marker='x')
    plt.title(f"{name} — Predicted vs Actual (First 100 Samples)")
    plt.xlabel("Time Step")
    plt.ylabel("Traffic Volume")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()