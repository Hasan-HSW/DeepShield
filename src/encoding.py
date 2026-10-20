"""
encoding.py
-----------
Converts text/categorical data into numbers for the LSTM:

1. Maps raw CICIDS2017 labels to the 6 target classes (config.LABEL_MAP),
   falling back to "Other" for anything unmapped rather than dropping it.
2. Selects the configured feature columns.
3. One-hot encodes categorical features (Protocol).
4. Label-encodes + one-hot encodes the target for softmax training.

The fitted LabelEncoder is saved to disk so training and inference always
agree on which index corresponds to which class.
"""

import logging

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from src import config

logger = logging.getLogger(__name__)


def map_single_label(raw: str) -> str:
    """
    Maps one raw CICIDS2017 label to a DeepShield target class.

    Tries an exact match against config.LABEL_MAP first. Falls back to a
    prefix check for "Web Attack ..." labels: the official CSVs store the
    separator between "Web Attack" and its sub-type (Brute Force / XSS /
    Sql Injection) as a mangled byte that decodes to U+FFFD regardless of
    file encoding (the original byte is unrecoverable), so an exact-string
    dict lookup never matches those rows. Since every Web Attack sub-type
    maps to the same "Web Attack" target class anyway, a prefix check gets
    the right answer without needing the exact separator character.
    Anything else unmapped becomes config.DEFAULT_LABEL ("Other") instead
    of being dropped, so rare attack types aren't silently lost.
    """
    raw = raw.strip()
    if raw in config.LABEL_MAP:
        return config.LABEL_MAP[raw]
    if raw.startswith("Web Attack"):
        return "Web Attack"
    return config.DEFAULT_LABEL


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps the raw Label column to DeepShield's 6 target classes.
    Anything not in config.LABEL_MAP becomes config.DEFAULT_LABEL ("Other")
    instead of being dropped, so rare attack types aren't silently lost.
    """
    df = df.copy()
    df[config.LABEL_COLUMN] = df[config.LABEL_COLUMN].str.strip()
    df[config.LABEL_COLUMN] = df[config.LABEL_COLUMN].apply(map_single_label)
    logger.info(f"Label distribution after mapping:\n{df[config.LABEL_COLUMN].value_counts()}")
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps only the configured feature columns plus the label column."""
    missing = [c for c in config.SELECTED_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(
            f"Expected feature columns not found in CSV: {missing}. "
            f"Check the file matches the CICIDS2017 schema."
        )
    cols = config.SELECTED_FEATURES + [config.LABEL_COLUMN]
    return df[cols].copy()


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encodes categorical feature columns (e.g. Protocol)."""
    df = df.copy()
    return pd.get_dummies(df, columns=config.CATEGORICAL_FEATURES)


def save_feature_columns(columns) -> None:
    """
    Saves the exact list/order of feature columns produced during training
    (after one-hot encoding). Required because a new CSV at inference time
    may contain a different subset of categorical values (e.g. only one
    Protocol type instead of several), which makes pd.get_dummies()
    produce a different set of columns than training did. Without
    reindexing to this saved list, scaler.transform() would receive a
    mismatched number of features and crash.
    """
    joblib.dump(list(columns), config.FEATURE_COLUMNS_PATH)
    logger.info(f"Saved feature column list to {config.FEATURE_COLUMNS_PATH}")


def load_feature_columns():
    """Loads the training-time feature column list."""
    return joblib.load(config.FEATURE_COLUMNS_PATH)


def align_features_to_training_columns(X_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindexes X_df to exactly match the columns seen during training, in
    the same order. Columns missing at inference (e.g. a Protocol value
    that didn't appear in this upload) are filled with 0. Extra/unexpected
    columns are dropped. This is what prevents the column-mismatch crash
    in scaler.transform() during /predict.
    """
    training_columns = load_feature_columns()
    return X_df.reindex(columns=training_columns, fill_value=0)


def fit_label_encoder() -> LabelEncoder:
    """Fits a LabelEncoder on the fixed target class list and saves it."""
    encoder = LabelEncoder()
    encoder.fit(config.TARGET_CLASSES)
    joblib.dump(encoder, config.LABEL_ENCODER_PATH)
    logger.info(f"Saved label encoder to {config.LABEL_ENCODER_PATH}")
    return encoder


def load_label_encoder() -> LabelEncoder:
    """Loads the previously fitted LabelEncoder (used at inference time)."""
    return joblib.load(config.LABEL_ENCODER_PATH)


def encode_labels(labels: pd.Series, encoder: LabelEncoder):
    """Converts string labels to one-hot vectors for softmax training."""
    integer_labels = encoder.transform(labels)
    return to_categorical(integer_labels, num_classes=len(config.TARGET_CLASSES))


def encoding_pipeline(df: pd.DataFrame, fit_encoder: bool = True):
    """
    Full encoding sequence.

    Args:
        df: cleaned DataFrame.
        fit_encoder: True for training (fits + saves new encoder),
            False for inference (loads the existing one).

    Returns:
        (X_df, y_onehot, encoder)
    """
    df = map_labels(df)
    df = select_features(df)
    df = encode_categorical_features(df)

    labels = df[config.LABEL_COLUMN]
    X_df = df.drop(columns=[config.LABEL_COLUMN])

    encoder = fit_label_encoder() if fit_encoder else load_label_encoder()
    y_onehot = encode_labels(labels, encoder)

    if fit_encoder:
        save_feature_columns(X_df.columns)

    return X_df, y_onehot, encoder


def encode_features_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    For unlabeled inference data (no Label column at all): selects and
    one-hot encodes features without touching labels.
    """
    missing = [c for c in config.SELECTED_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Expected feature columns not found in CSV: {missing}")
    X_df = df[config.SELECTED_FEATURES].copy()
    return encode_categorical_features(X_df)
