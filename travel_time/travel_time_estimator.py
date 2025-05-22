import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from math import radians, sin, cos, sqrt, atan2
from tensorflow.keras.models import load_model

# === Load Required Resources ===
model = load_model("training/models/lstm_scat_model.keras")
scalers = joblib.load("processed_data/scalers_by_scat.joblib")
dow_encoder = joblib.load("processed_data/onehot_encoder_dow.joblib")
scat_lookup = joblib.load("processed_data/scat_id_lookup.joblib")
inverse_scat_lookup = {v: k for k, v in scat_lookup.items()}

# Load traffic volume dataset with lat/lon
df = pd.read_csv("processed_data/scat_volume_with_coords_and_neighbours.csv")
df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
df["day_of_week"] = df["datetime"].dt.weekday
df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute

# === Haversine Distance ===
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# === Flow → Speed Conversion ===
def flow_to_speed(flow):
    if flow <= 351:
        return 60  # km/h
    a, b, c = -1.4648375, 93.75, -flow
    disc = b**2 - 4*a*c
    if disc < 0:
        return 20
    sqrt_disc = sqrt(disc)
    root1 = (-b + sqrt_disc) / (2*a)
    root2 = (-b - sqrt_disc) / (2*a)
    return min(root1, root2)  # lower root → congested zone

# === Predict Volume at a SCAT Site ===
def predict_volume(model, scat_number, datetime_str):
    scat_number = int(scat_number)
    dt = pd.to_datetime(datetime_str)
    df_scat = df[df['scat_number'] == scat_number]

    # Build average volume map
    avg_volume_map = (
        df_scat.groupby(['day_of_week', 'minute_of_day'])['volume']
        .mean()
        .reset_index()
        .set_index(['day_of_week', 'minute_of_day'])
    )

    # SCAT ID and scaler
    scat_id = scat_lookup[scat_number]
    scaler = scalers[scat_number]

    # Build input sequence
    sequence = []
    for i in range(8, 0, -1):
        step_time = dt - timedelta(minutes=15 * i)
        dow = step_time.weekday()
        minute = step_time.hour * 60 + step_time.minute
        time_norm = minute / 1440.0

        try:
            avg_volume = float(avg_volume_map.loc[(dow, minute)].iloc[0])
        except (KeyError, IndexError):
            avg_volume = float(df_scat["volume"].mean())

        vol_scaled = scaler.transform([[avg_volume]])[0][0]
        dow_df = pd.DataFrame([[dow]], columns=["day_of_week"])
        dow_onehot = dow_encoder.transform(dow_df)[0]
        feature = np.concatenate([[vol_scaled], dow_onehot, [time_norm]])
        sequence.append(feature)

    sequence = np.array(sequence).reshape(1, 8, 9)
    scat_input = np.array([[scat_id]])

    pred_scaled = model.predict([sequence, scat_input], verbose=0)[0][0]
    pred_volume = scaler.inverse_transform([[pred_scaled]])[0][0]
    return pred_volume

# === Final Function: Estimate Travel Time ===
# === Final Function: Estimate Travel Time ===
def estimate_travel_time(from_scat, to_scat, datetime_str, model):
    from_scat = int(from_scat)
    to_scat = int(to_scat)
    dt = pd.to_datetime(datetime_str)

    print(f"\nEstimating travel time from SCAT {from_scat} to SCAT {to_scat} at {datetime_str}")

    # Step 1: Get coordinates
    from_row = df[df["scat_number"] == from_scat].iloc[0]
    to_row = df[df["scat_number"] == to_scat].iloc[0]
    lat1, lon1 = from_row["NB_LATITUDE"], from_row["NB_LONGITUDE"]
    lat2, lon2 = to_row["NB_LATITUDE"], to_row["NB_LONGITUDE"]

    distance_km = haversine(lat1, lon1, lat2, lon2)
    print(f"Distance between SCAT {from_scat} and {to_scat}: {distance_km:.3f} km")

    # Step 2: Predict volume at FROM_SCAT (per assignment spec)
    volume = predict_volume(model, from_scat, dt)
    print(f"Predicted volume at SCAT {from_scat} at {datetime_str}: {volume:.2f} cars (per 15 mins)")

    flow = volume * 4  # convert to cars/hour
    print(f"Converted flow: {flow:.2f} cars/hour")

    # Step 3: Convert flow → speed
    speed = flow_to_speed(flow)
    print(f"Estimated traffic speed: {speed:.2f} km/h")

    # Step 4: Travel time = distance/speed + 30 sec
    travel_time_sec = (distance_km / speed) * 3600 + 30
    print(f"Estimated travel time: {travel_time_sec:.2f} seconds")

    return round(travel_time_sec, 2)
