from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from math import sqrt
import os
import sys

# Add preprocessing path
path = os.path.abspath("../data_processing")
sys.path.append(path)

import data_preprocessing

# Create directory to save models
os.makedirs("models", exist_ok=True)

# Load preprocessed training data
X_train, X_test, y_train, y_test = (
    data_preprocessing.X_train,
    data_preprocessing.X_test,
    data_preprocessing.y_train,
    data_preprocessing.y_test
)

# ---------------------- LSTM MODEL ----------------------
model_lstm = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)
pred_lstm = model_lstm.predict(X_test)
model_lstm.save("models/lstm_model.keras")

# ---------------------- GRU MODEL ----------------------
model_gru = Sequential([
    GRU(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])
model_gru.compile(optimizer='adam', loss='mse')
model_gru.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)
pred_gru = model_gru.predict(X_test)
model_gru.save("models/gru_model.keras")

# ---------------------- DENSE NN MODEL ----------------------
model_dense = Sequential([
    Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
model_dense.compile(optimizer='adam', loss='mse')
model_dense.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)
pred_dense = model_dense.predict(X_test)
model_dense.save("models/dense_model.keras")

# ---------------------- EVALUATION ----------------------
# Calculate and save all metrics
with open("models/model_evaluation.txt", "w") as f:
    def log_metrics(name, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        f.write(f"{name} MSE: {mse:.6f}\n")
        f.write(f"{name} RMSE: {rmse:.6f}\n")
        f.write(f"{name} MAE: {mae:.6f}\n\n")

    log_metrics("LSTM", y_test, pred_lstm)
    log_metrics("GRU", y_test, pred_gru)
    log_metrics("Dense NN", y_test, pred_dense)

print("✅ Models trained and evaluation results saved to models/model_evaluation.txt")
