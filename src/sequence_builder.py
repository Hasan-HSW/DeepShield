"""
sequence_builder.py
--------------------
Converts row-wise (2D) flow data into the 3D (samples, timesteps, features)
shape an LSTM requires, using a sliding window over consecutive flows.
Each window's label is taken from its LAST flow.
"""

import logging

import numpy as np

from src import config

logger = logging.getLogger(__name__)


def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    timesteps: int = config.TIMESTEPS,
    stride: int = config.SEQUENCE_STRIDE,
):
    """
    Slides a window of size `timesteps` across X (and matching y).

    Returns:
        X_seq: shape (num_sequences, timesteps, num_features)
        y_seq: shape (num_sequences, num_classes) — label of each window's last flow
    """
    num_rows, num_features = X.shape
    sequences_X, sequences_y = [], []

    for start in range(0, num_rows - timesteps + 1, stride):
        end = start + timesteps
        sequences_X.append(X[start:end])
        sequences_y.append(y[end - 1])

    X_seq = np.array(sequences_X)
    y_seq = np.array(sequences_y)

    logger.info(
        f"Built {X_seq.shape[0]} sequences of shape ({timesteps}, {num_features}) "
        f"from {num_rows} flows"
    )
    return X_seq, y_seq


def build_single_sequence_for_inference(X: np.ndarray, timesteps: int = config.TIMESTEPS):
    """
    Prepares one batch for real-time prediction. If fewer than `timesteps`
    flows are available, the earliest flow is repeated to pad the window
    (avoids introducing fake all-zero flows).
    """
    if len(X) < timesteps:
        pad_count = timesteps - len(X)
        padding = np.repeat(X[0:1], pad_count, axis=0)
        X = np.vstack([padding, X])
    else:
        X = X[-timesteps:]

    return np.expand_dims(X, axis=0)  # (1, timesteps, num_features)
