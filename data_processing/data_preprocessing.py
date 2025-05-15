import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load data
df = pd.read_csv("../processed_data/scats_volume_flat.csv")

# Step 2: Combine date and time for sorting and indexing
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df['day_of_week'] = df['datetime'].dt.weekday  # 0 = Monday
df['minute_of_day'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
df['time_norm'] = df['minute_of_day'] / 1440.0  # Normalize: 0–1439 min in day

# Step 3: Sort by SCAT and datetime
df = df.sort_values(by=['scat_number', 'datetime'])

# Step 4: One-hot encode day_of_week
encoder = OneHotEncoder(sparse_output=False, categories='auto')
dow_encoded = encoder.fit_transform(df[['day_of_week']])
df['dow_encoded'] = list(dow_encoded)

# Step 5: Group by SCAT
X_all = []
y_all = []
WINDOW_SIZE = 8
scalers = {}

for scat_id, group in df.groupby('scat_number'):
    group = group.copy()
    volumes = group['volume'].values.reshape(-1, 1)
    dow_vectors = np.vstack(group['dow_encoded'].values)  # shape: (N, 7)
    time_vector = group['time_norm'].values.reshape(-1, 1)  # shape: (N, 1)

    # Normalize volumes
    scaler = MinMaxScaler()
    volumes_scaled = scaler.fit_transform(volumes)
    scalers[scat_id] = scaler

    # Combine features: [DOW (7), time (1)] → shape (N, 8)
    full_feature_seq = np.hstack((dow_vectors, time_vector))

    for i in range(len(full_feature_seq) - WINDOW_SIZE):
        window = full_feature_seq[i:i + WINDOW_SIZE]  # shape (8, 8)
        X_all.append(window)
        y_all.append(volumes_scaled[i + WINDOW_SIZE])  # predict next volume

# Convert to arrays
X_all = np.array(X_all)  # (samples, 8, 8)
y_all = np.array(y_all)

# Save scalers and encoder
joblib.dump(scalers, "../processed_data/scalers_by_scat.joblib")
joblib.dump(encoder, "../processed_data/onehot_encoder_dow.joblib")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, shuffle=False
)

print(f"X_train shape: {X_train.shape}")  # (samples, 8, 8)
print(f"y_train shape: {y_train.shape}")
