"""
pipeline.py
-----------
Orchestrates the full preprocessing sequence for TRAINING:

    all CSVs in data/ -> clean -> encode -> split -> scale -> sequence

and a separate, lighter path for INFERENCE on one new CSV file.
"""

import logging

from sklearn.model_selection import train_test_split

from src import cleaning, config, encoding, scaling, sequence_builder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training_pipeline():
    """
    Runs the complete preprocessing pipeline for training. Loads every CSV
    in config.DATA_DIR, fits and saves the encoder + scaler.

    Returns:
        dict with X_train/y_train/X_val/y_val/X_test/y_test (LSTM-ready 3D
        arrays), plus the fitted label_encoder and scaler.
    """
    logger.info("Step 1/5: Loading and cleaning all CSVs in data/ ...")
    df = cleaning.clean_pipeline_from_folder()

    total_rows = len(df)
    raw_label_distribution = df[config.LABEL_COLUMN].value_counts().to_dict()

    logger.info("Step 2/5: Encoding labels and categorical features...")
    mapped_label_distribution = (
        encoding.map_labels(df)[config.LABEL_COLUMN].value_counts().to_dict()
    )
    X_df, y_onehot, label_encoder = encoding.encoding_pipeline(df, fit_encoder=True)
    num_features = X_df.shape[1]

    logger.info("Step 3/5: Splitting into train/val/test...")
    X_train_df, X_temp_df, y_train, y_temp = train_test_split(
        X_df, y_onehot,
        test_size=(config.TEST_SIZE + config.VAL_SIZE),
        random_state=config.RANDOM_STATE,
        stratify=y_onehot,
    )
    relative_val_size = config.VAL_SIZE / (config.TEST_SIZE + config.VAL_SIZE)
    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_temp_df, y_temp,
        test_size=(1 - relative_val_size),
        random_state=config.RANDOM_STATE,
        stratify=y_temp,
    )

    logger.info("Step 4/5: Scaling features (fit on train only)...")
    scaled = scaling.scaling_pipeline(X_train_df, X_val_df, X_test_df)

    logger.info("Step 5/5: Building LSTM sequences...")
    X_train_seq, y_train_seq = sequence_builder.build_sequences(scaled["train"], y_train)
    X_val_seq, y_val_seq = sequence_builder.build_sequences(scaled["val"], y_val)
    X_test_seq, y_test_seq = sequence_builder.build_sequences(scaled["test"], y_test)

    logger.info(
        f"Final shapes -> train: {X_train_seq.shape}, val: {X_val_seq.shape}, "
        f"test: {X_test_seq.shape}"
    )

    return {
        "X_train": X_train_seq, "y_train": y_train_seq,
        "X_val": X_val_seq, "y_val": y_val_seq,
        "X_test": X_test_seq, "y_test": y_test_seq,
        "label_encoder": label_encoder,
        "scaler": scaled["scaler"],
        "dataset_path": config.DATA_DIR,
        "total_rows": total_rows,
        "num_features": num_features,
        "raw_label_distribution": raw_label_distribution,
        "mapped_label_distribution": mapped_label_distribution,
    }


def run_inference_pipeline(csv_path: str):
    """
    Preprocesses ONE new CSV file for prediction, reusing the encoder/scaler
    saved during training. Handles both labeled and unlabeled input files.

    Returns:
        X_seq: 3D array ready for model.predict()
    """
    df = cleaning.clean_pipeline_single_file(csv_path)

    if config.LABEL_COLUMN in df.columns:
        X_df, _, _ = encoding.encoding_pipeline(df, fit_encoder=False)
    else:
        X_df = encoding.encode_features_only(df)

    # Reindex to the exact columns seen during training — a new upload may
    # have a different set of one-hot Protocol columns (e.g. only one
    # protocol present), which would otherwise crash the scaler/model.
    X_df = encoding.align_features_to_training_columns(X_df)

    scaler = scaling.load_scaler()
    X_scaled = scaling.transform(X_df, scaler)

    return sequence_builder.build_single_sequence_for_inference(X_scaled)
