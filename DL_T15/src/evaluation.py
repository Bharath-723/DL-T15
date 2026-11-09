"""Window-level threshold and metric helpers used by the submitted notebook.

The notebook's FP/FN and rupee-cost values are window-level counts, not merged
machine-fault/alarm events. See docs/NOTEBOOK_AUDIT.md before using them as
operational plant costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score
)

MISSED_FAULT_COST_RUPEES = 400_000
FALSE_ALARM_COST_RUPEES = 2_000


def find_best_f1_threshold(errors, labels, n_candidates: int = 300):
    """Search candidate percentiles (50th to 99.9th) for maximum window F1."""
    errors = np.asarray(errors, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if errors.ndim != 1 or labels.ndim != 1 or len(errors) != len(labels):
        raise ValueError("errors and labels must be aligned 1-D arrays")
    if not len(errors):
        raise ValueError("Cannot select a threshold from empty arrays")
    thresholds = np.unique(np.percentile(errors, np.linspace(50, 99.9, n_candidates)))
    rows = []
    for threshold in thresholds:
        predictions = (errors > threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "precision": precision_score(labels, predictions, zero_division=0),
            "recall": recall_score(labels, predictions, zero_division=0),
            "f1": f1_score(labels, predictions, zero_division=0),
        })
    table = pd.DataFrame(rows)
    return table.loc[table["f1"].idxmax()], table


def optimise_window_cost_threshold(
    errors,
    labels,
    missed_fault_cost: int = MISSED_FAULT_COST_RUPEES,
    false_alarm_cost: int = FALSE_ALARM_COST_RUPEES,
):
    """Minimise window-level FP/FN cost on validation data."""
    errors = np.asarray(errors, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if errors.ndim != 1 or labels.ndim != 1 or len(errors) != len(labels):
        raise ValueError("errors and labels must be aligned 1-D arrays")
    if not len(errors):
        raise ValueError("Cannot optimise a threshold on empty arrays")
    candidates = np.unique(np.concatenate(([0.0], errors, [errors.max() + 1e-10])))
    best = None
    for threshold in candidates:
        predictions = (errors >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        cost = missed_fault_cost * fn + false_alarm_cost * fp
        candidate = {
            "threshold": float(threshold),
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "window_cost_rupees": int(cost),
        }
        if best is None or candidate["window_cost_rupees"] < best["window_cost_rupees"]:
            best = candidate
    return best


def evaluate_window_predictions(labels, predictions, model_name: str = "model"):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    if labels.ndim != 1 or predictions.ndim != 1 or len(labels) != len(predictions):
        raise ValueError("labels and predictions must be aligned 1-D arrays")
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "Model": model_name,
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "window_cost_rupees": int(
            MISSED_FAULT_COST_RUPEES * fn + FALSE_ALARM_COST_RUPEES * fp
        ),
    }


def count_alarm_runs(binary_predictions):
    """Count starts of contiguous positive windows; this is not event matching."""
    values = np.asarray(binary_predictions, dtype=int)
    if values.ndim != 1:
        raise ValueError("predictions must be one-dimensional")
    starts = (values == 1) & (np.r_[0, values[:-1]] == 0)
    return int(starts.sum())
