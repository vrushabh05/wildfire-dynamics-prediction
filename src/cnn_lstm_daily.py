"""CNN-LSTM that predicts tomorrow's fire map for California from the last 7 days.

Data: NASA FIRMS MODIS active-fire detections for the United States, 2023
(`modis_2023_United_States.csv`, not included; download it from NASA FIRMS).
Each day becomes a 100x100 grid of summed fire brightness, min-max normalized.
"""
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

CSV = "modis_2023_United_States.csv"
LAT_MIN, LAT_MAX = 32.0, 42.0
LON_MIN, LON_MAX = -125.0, -114.0
GRID = (100, 100)
SEQ_LEN = 7


def build_daily_grids(csv_path=CSV):
    df = pd.read_csv(csv_path)
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df_ca = df[(df["latitude"] >= LAT_MIN) & (df["latitude"] <= LAT_MAX)
               & (df["longitude"] >= LON_MIN) & (df["longitude"] <= LON_MAX)]
    lat_bins = np.linspace(LAT_MIN, LAT_MAX, GRID[0] + 1)
    lon_bins = np.linspace(LON_MIN, LON_MAX, GRID[1] + 1)

    grids = []
    for _, group in sorted(df_ca.groupby("acq_date"), key=lambda kv: kv[0]):
        hist, _, _ = np.histogram2d(group["latitude"], group["longitude"],
                                    bins=[lat_bins, lon_bins], weights=group["brightness"])
        if hist.max() != hist.min():
            hist = (hist - hist.min()) / (hist.max() - hist.min())
        if hist.sum() > 0:                       # drop days with no detections
            grids.append(hist)
    return grids


def make_sequences(grids, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(len(grids) - seq_len):
        X.append(grids[i:i + seq_len])
        y.append(grids[i + seq_len])
    return np.array(X), np.array(y)


def build_model(input_shape):
    model = models.Sequential([
        layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                               input_shape=input_shape),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation="relu", padding="same")),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Flatten()),
        layers.LSTM(64),
        layers.Dense(GRID[0] * GRID[1], activation="sigmoid"),
        layers.Reshape(GRID),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


if __name__ == "__main__":
    grids = build_daily_grids()
    X, y = make_sequences(grids)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    model = build_model(X_train.shape[1:])
    model.summary()
    history = model.fit(X_train, y_train, epochs=5, batch_size=3, validation_split=0.3)

    loss, mae = model.evaluate(X_test, y_test)
    print("Test MSE:", loss)
    print("Test MAE:", mae)

    idx = random.randint(0, len(X_test) - 1)
    pred = model.predict(X_test[idx:idx + 1])[0]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, img, title in zip(ax, [y_test[idx], pred, np.abs(pred - y_test[idx])],
                             ["Actual", "Predicted", "Absolute error"]):
        im = a.imshow(img, cmap="hot" if title != "Absolute error" else "viridis")
        a.set_title(title)
        fig.colorbar(im, ax=a)
    plt.tight_layout()
    plt.savefig("sample_prediction.png", dpi=150)
