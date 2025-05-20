from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Flatten, Embedding, Concatenate, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping
from math import sqrt
import numpy as np
import os
import sys
import joblib

# ---------------- Load preprocessed data ----------------
path = os.path.abspath("../data_processing")
sys.path.append(path)
import data_preprocessing

X_train = data_preprocessing.X_train_seq
X_test = data_preprocessing.X_test_seq
scat_train = data_preprocessing.scat_train
scat_test = data_preprocessing.scat_test
y_train = data_preprocessing.y_train
y_test = data_preprocessing.y_test

# ---------------- Load SCAT metadata ----------------
scalers = joblib.load("../processed_data/scalers_by_scat.joblib")
scat_id_lookup = joblib.load("../processed_data/scat_id_lookup.joblib")
num_scat_ids = len(scat_id_lookup)

# ---------------- Common model components ----------------
def build_scat_embedding(scat_input, sequence_len):
    scat_embed = Embedding(input_dim=num_scat_ids, output_dim=8)(scat_input)  # (None, 1, 8)
    scat_embed = Flatten()(scat_embed)                                        # (None, 8)
    scat_embed = RepeatVector(sequence_len)(scat_embed)                       # (None, sequence_len, 8)
    return scat_embed

def build_model(model_type):
    sequence_input = Input(shape=(X_train.shape[1], X_train.shape[2]), name="sequence_input")
    scat_input = Input(shape=(1,), name="scat_input")
    scat_embed = build_scat_embedding(scat_input, X_train.shape[1])
    merged = Concatenate(axis=-1)([sequence_input, scat_embed])  # (None, 8, 17)

    if model_type == "LSTM":
        x = LSTM(128, return_sequences=True)(merged)
        x = LSTM(64)(x)
    elif model_type == "GRU":
        x = GRU(128, return_sequences=True)(merged)
        x = GRU(64)(x)
    elif model_type == "Dense":
        x = Flatten()(merged)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
    else:
        raise ValueError("Invalid model_type")

    output = Dense(1)(x)
    model = Model(inputs=[sequence_input, scat_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# ---------------- Training loop ----------------
models = {}
predictions = {}
model_types = ["LSTM", "GRU", "Dense"]
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

for mtype in model_types:
    print(f"\nTraining {mtype} model...")
    model = build_model(mtype)
    model.fit(
        [X_train, scat_train],
        y_train,
        validation_data=([X_test, scat_test], y_test),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    model.save(f"models/{mtype.lower()}_scat_model.keras")
    y_pred_scaled = model.predict([X_test, scat_test])
    predictions[mtype] = y_pred_scaled
    models[mtype] = model

# ---------------- Evaluation ----------------
y_test = y_test.reshape(-1, 1)

with open("models/model_evaluation.txt", "w") as f:
    def log_metrics(name, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        f.write(f"{name} MSE: {mse:.6f}\n")
        f.write(f"{name} RMSE: {rmse:.6f}\n")
        f.write(f"{name} MAE: {mae:.6f}\n\n")

    for name, y_pred_scaled in predictions.items():
        scaler = scalers[list(scalers.keys())[0]]  # For now, use the same scaler for all
        y_true = scaler.inverse_transform(y_test)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        log_metrics(f"{name} + SCAT + Volume", y_true, y_pred)
        
print("All models trained and saved. Evaluation saved to models/model_evaluation.txt")
