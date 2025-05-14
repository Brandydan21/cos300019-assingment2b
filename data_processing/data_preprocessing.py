import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load data
df = pd.read_csv("../processed_data/scats_volume_flat.csv")

# Step 2: Combine date and time for sorting and indexing
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

# Step 3: Sort by SCAT and datetime
df = df.sort_values(by=['scat_number', 'datetime'])

# Step 4: Group by each SCAT and prepare sequences
X_all = []
y_all = []
WINDOW_SIZE = 8

scalers = {}

for scat_id, group in df.groupby('scat_number'):
    group = group.copy()
    volumes = group['volume'].values.reshape(-1, 1)

    # Step 5: Normalize the volume values
    scaler = MinMaxScaler()
    volumes_scaled = scaler.fit_transform(volumes)
    scalers[scat_id] = scaler

    # Step 6: Create sliding windows
    for i in range(len(volumes_scaled) - WINDOW_SIZE):
        X_all.append(volumes_scaled[i:i + WINDOW_SIZE])
        y_all.append(volumes_scaled[i + WINDOW_SIZE])

# Step 7: Convert to numpy arrays
X_all = np.array(X_all)
y_all = np.array(y_all)

# Step 8: Save the scaler dictionary (optional)
joblib.dump(scalers, "../processed_data/scalers_by_scat.joblib")

# Step 9: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, shuffle=False)

# X_train: shape (samples, time_steps, 1) — ready for LSTM/GRU
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
