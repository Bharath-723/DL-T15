"""Small plotting helpers for the Team 15 anomaly-detection workflow."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def plot_loss(history, title="Autoencoder training loss", output_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history.get("loss", []), label="Training loss")
    if "val_loss" in history.history:
        ax.plot(history.history["val_loss"], label="Validation loss")
    ax.set(title=title, xlabel="Epoch", ylabel="MSE")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def plot_reconstruction_errors(errors, threshold=None, title="Reconstruction error", output_path=None):
    errors = np.asarray(errors, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(errors, linewidth=0.8, label="Reconstruction error")
    if threshold is not None:
        ax.axhline(threshold, linestyle="--", label=f"Threshold = {threshold:.6g}")
    ax.set(title=title, xlabel="Window index", ylabel="Mean squared reconstruction error")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def plot_precision_recall_curve(
    errors,
    labels,
    operating_thresholds=None,
    title="Validation precision-recall curve",
    output_path=None,
):
    """Plot a PR curve from anomaly scores and optionally mark chosen thresholds.

    `operating_thresholds` should map labels to scalar thresholds, e.g.
    {"Best F1": 0.04, "Cost-optimal": 0.02}.
    """
    scores = np.asarray(errors, dtype=float)
    y_true = np.asarray(labels, dtype=int)
    if scores.ndim != 1 or y_true.ndim != 1 or len(scores) != len(y_true):
        raise ValueError("errors and labels must be aligned 1-D arrays")

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(recall, precision, label="Precision-recall curve")
    if operating_thresholds:
        for label, threshold in operating_thresholds.items():
            predictions = (scores >= float(threshold)).astype(int)
            tp = np.sum((predictions == 1) & (y_true == 1))
            fp = np.sum((predictions == 1) & (y_true == 0))
            fn = np.sum((predictions == 0) & (y_true == 1))
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            ax.scatter([r], [p], s=55, label=f"{label} (t={float(threshold):.4g})")
    ax.set(
        title=title,
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig
