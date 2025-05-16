from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# ---------------- Load model + encoders ----------------
model = load_model("../training/models/lstm_scat_model.keras")
scalers = joblib.load("../processed_data/scalers_by_scat.joblib")
dow_encoder = joblib.load("../processed_data/onehot_encoder_dow.joblib")
scat_lookup = joblib.load("../processed_data/scat_id_lookup.joblib")
inverse_scat_lookup = {v: k for k, v in scat_lookup.items()}

# ---------------- Input: SCAT + target datetime ----------------
target_scat = 970
target_datetime = pd.to_datetime("2006-11-14 23:30")

# ---------------- Load historical data ----------------
df = pd.read_csv("../processed_data/scats_volume_flat.csv")
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df['day_of_week'] = df['datetime'].dt.weekday
df['minute_of_day'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute

# Filter for SCAT
df_scat = df[df['scat_number'] == target_scat]

# ---------------- Compute average volume map ----------------
avg_volume_map = (
    df_scat.groupby(['day_of_week', 'minute_of_day'])['volume']
    .mean()
    .reset_index()
    .set_index(['day_of_week', 'minute_of_day'])
)

# ---------------- Build past 8 steps using avg volume ----------------
sequence = []
scat_id = scat_lookup[target_scat]
scaler = scalers[target_scat]

for i in range(8, 0, -1):
    dt = target_datetime - timedelta(minutes=15 * i)
    dow = dt.weekday()
    minute = dt.hour * 60 + dt.minute
    time_norm = minute / 1440.0

    # Get average volume for this time
    try:
        avg_volume = float(avg_volume_map.loc[(dow, minute)])
    except KeyError:
        avg_volume = float(df_scat['volume'].mean())  # fallback to general mean

    vol_scaled = scaler.transform([[avg_volume]])[0][0]
    dow_onehot = dow_encoder.transform([[dow]])[0]
    feature_vector = np.concatenate([[vol_scaled], dow_onehot, [time_norm]])
    sequence.append(feature_vector)

sequence = np.array(sequence).reshape(1, 8, 9)  # (1, 8, features)
scat_input = np.array([[scat_id]])  # (1, 1)

# ---------------- Predict ----------------
pred_scaled = model.predict([sequence, scat_input])[0][0]
pred_volume = scaler.inverse_transform([[pred_scaled]])[0][0]

# ---------------- Output ----------------
print(f"Predicted volume at {target_datetime} for SCAT {target_scat} → {pred_volume:.2f}")
