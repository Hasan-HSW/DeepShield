"""
scaling.py
----------
MinMaxScaler normalization. The scaler is fit ONLY on the training set
and then used to transform train/val/test separately, to avoid leaking
val/test statistics into the training process.
"""

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src import config

logger = logging.getLogger(__name__)


def fit_scaler(X_train: pd.DataFrame) -> MinMaxScaler:
    """Fits MinMaxScaler on training features only, and saves it to disk."""
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, config.SCALER_PATH)
    logger.info(f"Saved fitted scaler to {config.SCALER_PATH}")
    return scaler


def load_scaler() -> MinMaxScaler:
    """Loads the previously fitted scaler (used at inference time)."""
    return joblib.load(config.SCALER_PATH)


def transform(X: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
    """Applies an already-fitted scaler to a feature set."""
    return scaler.transform(X)


def scaling_pipeline(X_train, X_val=None, X_test=None):
    """
    Fits the scaler on X_train, transforms whichever of X_train/X_val/X_test
    are provided, and returns them in a dict along with the fitted scaler.
    """
    scaler = fit_scaler(X_train)
    result = {"train": transform(X_train, scaler), "scaler": scaler}

    if X_val is not None:
        result["val"] = transform(X_val, scaler)
    if X_test is not None:
        result["test"] = transform(X_test, scaler)

    return result
