"""
model.py
--------
Defines the LSTM architecture used to classify network traffic sequences
into one of the 6 target attack classes.
"""

from tensorflow import keras
from tensorflow.keras import layers

from src import config


def build_lstm_model(timesteps: int, num_features: int, num_classes: int) -> keras.Model:
    """
    Builds and compiles the LSTM classifier.

    Architecture: two stacked LSTM layers (with dropout between each) feed
    into a small dense head. This depth is enough to capture temporal
    attack patterns across a window of flows without being so large that
    it overfits CICIDS2017's relatively small minority classes.

    Args:
        timesteps: number of flows per input sequence.
        num_features: number of features per timestep.
        num_classes: number of output classes.

    Returns:
        A compiled keras.Model.
    """
    model = keras.Sequential([
        layers.Input(shape=(timesteps, num_features)),

        layers.LSTM(config.LSTM_UNITS, return_sequences=True),
        layers.Dropout(config.DROPOUT_RATE),

        layers.LSTM(config.LSTM_UNITS // 2),
        layers.Dropout(config.DROPOUT_RATE),

        layers.Dense(32, activation="relu"),
        layers.Dropout(config.DROPOUT_RATE / 2),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_callbacks(checkpoint_path: str):
    """Early stopping + best-model checkpoint + LR reduction on plateau."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]
