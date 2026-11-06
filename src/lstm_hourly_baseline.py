"""First iteration: an LSTM over hourly fire counts per 0.5-degree grid cell.

Data: NASA FIRMS MODIS 2023 US detections (`modis_2023_United_States.csv`,
no header row, not included).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

COLUMNS = ["latitude", "longitude", "brightness", "scan", "track", "acq_date", "acq_time",
           "satellite", "instrument", "confidence", "version", "bright_t31", "frp",
           "daynight", "type"]
SEQ_LEN = 6


def create_sequences(data, seq_length=SEQ_LEN):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


df = pd.read_csv("modis_2023_United_States.csv", names=COLUMNS, header=None)
df["acq_time"] = df["acq_time"].astype(str).str.zfill(4)
df["datetime"] = pd.to_datetime(df["acq_date"] + " " + df["acq_time"], format="%Y-%m-%d %H%M",
                                errors="coerce")
df = df.sort_values("datetime")

# 0.5-degree spatial bins, hourly time steps
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df["lat_bin"] = (df["latitude"] // 0.5) * 0.5
df["lon_bin"] = (df["longitude"] // 0.5) * 0.5
df["hour"] = df["datetime"].dt.floor("h")

agg = df.groupby(["hour", "lat_bin", "lon_bin"]).size().reset_index(name="fire_count")
pivot = agg.pivot_table(index="hour", columns=["lat_bin", "lon_bin"],
                        values="fire_count", fill_value=0).sort_index()
print(pivot.shape)                         # (3246, 2768) in the original run

data = pivot.values
scaler = MinMaxScaler()
X, y = create_sequences(scaler.fit_transform(data))
split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

model = Sequential([LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
                    Dense(y_train.shape[1])])
model.compile(optimizer="adam", loss="mse")
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test), scaler.inverse_transform(y_pred)))
print(f"RMSE: {rmse:.2f}")
