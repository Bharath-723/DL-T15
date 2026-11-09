"""Keras model builders matching Team 15's notebook architectures."""
from __future__ import annotations

from tensorflow.keras import layers, models


def build_dense_autoencoder(window_size: int = 12):
    """Dense baseline: 12 -> 8 -> 4 -> 2 -> 4 -> 8 -> 12."""
    model = models.Sequential([
        layers.Input(shape=(window_size,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(4, activation="relu"),
        layers.Dense(2, activation="relu"),
        layers.Dense(4, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(window_size, activation="sigmoid"),
    ], name="dense_autoencoder_baseline")
    model.compile(optimizer="adam", loss="mse")
    return model


def build_improved_dense_autoencoder(window_size: int = 12):
    """Narrow-bottleneck dense model: 12 -> 6 -> 3 -> 1 -> 3 -> 6 -> 12."""
    model = models.Sequential([
        layers.Input(shape=(window_size,)),
        layers.Dense(6, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(1, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(6, activation="relu"),
        layers.Dense(window_size, activation="sigmoid"),
    ], name="improved_dense_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def build_lstm_autoencoder(window_size: int = 12, n_features: int = 1):
    """Sequence model used in the submitted notebook."""
    model = models.Sequential([
        layers.Input(shape=(window_size, n_features)),
        layers.LSTM(32, activation="tanh"),
        layers.Dense(16, activation="relu"),
        layers.RepeatVector(window_size),
        layers.LSTM(32, activation="tanh", return_sequences=True),
        layers.TimeDistributed(layers.Dense(n_features, activation="sigmoid")),
    ], name="lstm_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model
