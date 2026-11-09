"""Data preparation helpers for Team 15's NAB machine-temperature experiment.

These utilities are extracted/refactored from the submitted notebook. Important:
the notebook's current training workflow filters training observations using NAB
anomaly-window labels. That conflicts with the assignment rule that labels are
reserved for validation/threshold tuning/testing. See docs/NOTEBOOK_AUDIT.md.
"""
from __future__ import annotations

from typing import Iterable, Sequence
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/numenta/NAB/master/data/"
    "realKnownCause/machine_temperature_system_failure.csv"
)
WINDOWS_URL = (
    "https://raw.githubusercontent.com/numenta/NAB/master/labels/"
    "combined_windows.json"
)
NAB_KEY = "realKnownCause/machine_temperature_system_failure.csv"


def load_temperature_csv(path: str | Path | None = None) -> pd.DataFrame:
    """Load and clean the NAB temperature CSV; download if path is omitted."""
    frame = pd.read_csv(path or DATA_URL)
    required = {"timestamp", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "value"])
    frame = frame.sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="first")
    return frame.reset_index(drop=True)


def load_nab_windows(path: str | Path | None = None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Load official NAB event windows for the machine-temperature file."""
    if path is None:
        import requests
        response = requests.get(WINDOWS_URL, timeout=30)
        response.raise_for_status()
        payload = response.json()
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if NAB_KEY not in payload:
        raise KeyError(f"{NAB_KEY!r} not found in NAB combined_windows.json")
    return [
        (pd.to_datetime(start), pd.to_datetime(end))
        for start, end in payload[NAB_KEY]
    ]


def attach_window_labels(
    frame: pd.DataFrame,
    windows: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    """Add `in_anomaly_window`; intended for validation and test evaluation."""
    result = frame.copy()
    mask = np.zeros(len(result), dtype=bool)
    timestamps = result["timestamp"]
    for start, end in windows:
        mask |= timestamps.between(start, end, inclusive="both").to_numpy()
    result["in_anomaly_window"] = mask
    return result


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split observations by time order, without shuffling."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must be less than 1")
    n = len(frame)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + validation_fraction))
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:val_end].copy(),
        frame.iloc[val_end:].copy(),
    )


def create_sequences(values: Iterable[float], window_size: int = 12) -> np.ndarray:
    """Convert a one-dimensional stream to overlapping fixed-length windows."""
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if len(array) < window_size:
        return np.empty((0, window_size), dtype=np.float32)
    return np.stack(
        [array[i:i + window_size] for i in range(len(array) - window_size + 1)]
    )


def create_window_labels(
    frame: pd.DataFrame,
    window_size: int = 12,
    label_column: str = "in_anomaly_window",
) -> np.ndarray:
    """Label a window anomalous when any observation intersects an NAB window."""
    if label_column not in frame:
        raise KeyError(f"Expected column {label_column!r}")
    labels = frame[label_column].astype(bool).to_numpy()
    if len(labels) < window_size:
        return np.empty(0, dtype=np.int8)
    return np.array(
        [int(labels[i:i + window_size].any())
         for i in range(len(labels) - window_size + 1)],
        dtype=np.int8,
    )


def make_normal_training_windows(
    train_frame: pd.DataFrame,
    scaled_column: str = "scaled_value",
    label_column: str = "in_anomaly_window",
    window_size: int = 12,
) -> np.ndarray:
    """Reproduce the notebook's label-filtered training construction.

    WARNING: retained only to document/reproduce the submitted notebook.
    Filtering the training data with NAB labels is not compliant with the brief's
    'labels only for threshold tuning and testing' rule. It also concatenates
    records across removed intervals before windowing. See the audit document.
    """
    if scaled_column not in train_frame or label_column not in train_frame:
        raise KeyError(f"Expected {scaled_column!r} and {label_column!r}")
    normal_rows = train_frame.loc[~train_frame[label_column].astype(bool)]
    return create_sequences(normal_rows[scaled_column].to_numpy(), window_size)
