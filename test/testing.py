from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib

# Load model and scalers
model = load_model("../training/models/lstm_model.keras")
scalers = joblib.load("../processed_data/scalers_by_scat.joblib")

# Load and prepare data
df = pd.read_csv("../processed_data/scats_volume_flat.csv")
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

# Choose SCATS site
scat_id = 970
site_df = df[df['scat_number'] == scat_id].sort_values('datetime')

# -------------------- SET CUSTOM TARGET TIME --------------------
# Example: predict volume after this datetime
target_datetime = pd.to_datetime("2006-10-31 18:30:00")

# Filter only history before the chosen time
history_df = site_df[site_df['datetime'] < target_datetime]

# Check and get last 8 volumes before that time
recent_volumes = history_df['volume'].values[-8:].reshape(-1, 1)
if len(recent_volumes) < 8:
    raise ValueError(f"Not enough history before {target_datetime} for prediction.")

# Scale input and reshape
scaler = scalers[scat_id]
X_input = scaler.transform(recent_volumes).reshape(1, 8, 1)

# Predict
pred_scaled = model.predict(X_input)
predicted_volume = scaler.inverse_transform(pred_scaled)[0][0]

# Print result
print(f"Predicted volume after {target_datetime} for SCAT {scat_id}: {predicted_volume:.2f}")
