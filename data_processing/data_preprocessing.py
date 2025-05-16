import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load data
df = pd.read_csv("../processed_data/scats_volume_flat.csv")

# Step 2: Extract features
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df['day_of_week'] = df['datetime'].dt.weekday  # 0 = Monday
df['minute_of_day'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
df['time_norm'] = df['minute_of_day'] / 1440.0

# Step 3: Sort and encode
df = df.sort_values(by=['scat_number', 'datetime']).reset_index(drop=True)

# One-hot encode day_of_week
dow_encoder = OneHotEncoder(sparse_output=False, categories='auto')
dow_encoded = dow_encoder.fit_transform(df[['day_of_week']])
df['dow_encoded'] = list(dow_encoded)

# Encode SCAT IDs as integers for embedding
scat_id_lookup = {scat: idx for idx, scat in enumerate(df['scat_number'].unique())}
df['scat_id_int'] = df['scat_number'].map(scat_id_lookup)

# Store mappings
joblib.dump(dow_encoder, "../processed_data/onehot_encoder_dow.joblib")
joblib.dump(scat_id_lookup, "../processed_data/scat_id_lookup.joblib")

# Prepare sequences
WINDOW_SIZE = 8
X_seq = []
scat_ids = []
y_all = []
scalers = {}

for scat_id, group in df.groupby('scat_number'):
    group = group.copy()
    group = group.reset_index(drop=True)

    # Volume & feature prep
    volumes = group['volume'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    volumes_scaled = scaler.fit_transform(volumes)
    scalers[scat_id] = scaler

    # Feature vectors
    volume_vector = volumes_scaled  # shape (N, 1)
    dow_vectors = np.vstack(group['dow_encoded'].values)  # (N, 7)
    time_vector = group['time_norm'].values.reshape(-1, 1)  # (N, 1)

    # Final per-time-step features = [volume, DOW one-hot, time_norm]
    feature_matrix = np.hstack((volume_vector, dow_vectors, time_vector))  # (N, 9)

    for i in range(len(group) - WINDOW_SIZE):
        X_seq.append(feature_matrix[i:i + WINDOW_SIZE])       # shape (8, 9)
        scat_ids.append(scat_id_lookup[scat_id])              # single int
        y_all.append(volumes_scaled[i + WINDOW_SIZE])         # predict next

# Convert to arrays
X_seq = np.array(X_seq)               # (samples, 8, 9)
scat_ids = np.array(scat_ids)         # (samples,)
y_all = np.array(y_all)               # (samples,)

# Save scalers
joblib.dump(scalers, "../processed_data/scalers_by_scat.joblib")

# Split data
X_train, X_test, scat_train, scat_test, y_train, y_test = train_test_split(
    X_seq, scat_ids, y_all, test_size=0.2, shuffle=False
)

# Make available for training script
X_train_seq = X_train
X_test_seq = X_test
scat_train = scat_train
scat_test = scat_test
y_train = y_train
y_test = y_test

print(f"X_train_seq shape: {X_train_seq.shape}")  # (samples, 8, 9)
print(f"scat_train shape: {scat_train.shape}")    # (samples,)
print(f"y_train shape: {y_train.shape}")          # (samples,)
