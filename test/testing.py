from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib

# ---------------- Load model, encoder, and scaler ----------------
model = load_model("../training/models/lstm_model.keras")
encoder = joblib.load("../processed_data/onehot_encoder_dow.joblib")
scalers = joblib.load("../processed_data/scalers_by_scat.joblib")
scaler = scalers[list(scalers.keys())[0]]  # Use first one (e.g., 970)

# ---------------- Define future datetime targets ----------------
future_datetimes = [
    "2006-11-21 15:30"
]

# ---------------- Convert to features ----------------
future_features = []
for dt_str in future_datetimes:
    dt = pd.to_datetime(dt_str)
    day = dt.weekday()  # Monday=0
    minutes_since_midnight = dt.hour * 60 + dt.minute
    time_of_day_norm = minutes_since_midnight / 1440  # Normalize to [0, 1]
    day_onehot = encoder.transform([[day]])[0]        # shape (7,)
    full_feature = np.append(day_onehot, time_of_day_norm)  # shape (8,)
    future_features.append(full_feature)

# ---------------- Build input sequence ----------------
if len(future_features) >= 8:
    input_sequence = np.array(future_features[:8])
else:
    input_sequence = np.tile(future_features[0], (8, 1))  # repeat if fewer than 8
input_sequence = input_sequence.reshape(1, 8, 8)

# ---------------- Predict recursively ----------------
predictions = []
for i, features in enumerate(future_features):
    pred_scaled = model.predict(input_sequence)[0][0]
    pred_volume = scaler.inverse_transform([[pred_scaled]])[0][0]
    predictions.append((future_datetimes[i], pred_volume))

    input_sequence = np.append(input_sequence[:, 1:, :], features.reshape(1, 1, 8), axis=1)

# ---------------- Output ----------------
print("Predicted volumes for given dates and times:")
for dt_str, volume in predictions:
    print(f"{dt_str} → {volume:.2f}")
