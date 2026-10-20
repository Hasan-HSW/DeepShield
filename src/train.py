"""
train.py
--------
End-to-end training script:

    1. Run the preprocessing pipeline (loads all CSVs in data/)
    2. Build the LSTM model
    3. Train with early stopping / checkpointing
    4. Evaluate on the held-out test set (accuracy, precision, recall,
       F1, confusion matrix)
    5. Save the final model to models/model.h5 and metrics to outputs/

Run from the project root:
    python -m src.train
"""

import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed — saving plots to file only
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src import cleaning, config, model as model_module, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluates the trained model on the test set and prints/saves
    accuracy, precision, recall, F1-score, and a confusion matrix image.
    This is the core of the project's V&V (Verification & Validation) report.
    """
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    class_names = label_encoder.classes_

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    summary = (
        f"Test Accuracy:  {accuracy:.4f}\n"
        f"Test Precision: {precision:.4f} (weighted)\n"
        f"Test Recall:    {recall:.4f} (weighted)\n"
        f"Test F1-score:  {f1:.4f} (weighted)\n\n"
        f"Classification Report:\n{report}\n"
    )
    print(summary)

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    with open(config.METRICS_REPORT_PATH, "w") as f:
        f.write(summary)
    logger.info(f"Saved metrics report to {config.METRICS_REPORT_PATH}")

    # Save confusion matrix as an image
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix — DeepShield LSTM")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(config.CONFUSION_MATRIX_PATH)
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot to {config.CONFUSION_MATRIX_PATH}")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main():
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    logger.info("Running preprocessing pipeline...")
    data = pipeline.run_training_pipeline()

    print("\n" + "=" * 70)
    print("Model is trained on REAL CIC-IDS-2017 dataset")
    print(f"Dataset path: {data['dataset_path']}")
    print(f"Total rows used for training (after cleaning): {data['total_rows']:,}")
    print(f"Number of input features (after encoding): {data['num_features']}")
    print("\nRaw label distribution (original CICIDS2017 labels):")
    for label, count in data["raw_label_distribution"].items():
        print(f"   {label}: {count:,}")
    print("\nTarget class distribution (after mapping to the 6 DeepShield classes):")
    for label, count in data["mapped_label_distribution"].items():
        print(f"   {label}: {count:,}")
    print("=" * 70 + "\n")

    timesteps = data["X_train"].shape[1]
    num_features = data["X_train"].shape[2]
    num_classes = len(config.TARGET_CLASSES)

    logger.info(f"Building LSTM model: timesteps={timesteps}, "
                f"features={num_features}, classes={num_classes}")
    model = model_module.build_lstm_model(timesteps, num_features, num_classes)
    model.summary()

    checkpoint_path = config.MODEL_PATH
    callbacks = model_module.get_callbacks(checkpoint_path)

    logger.info("Starting training...")
    model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
    )

    # ModelCheckpoint already saved the best weights to config.MODEL_PATH,
    # but we save again explicitly at the end for clarity / in case
    # training finished without triggering a new checkpoint.
    model.save(config.MODEL_PATH)
    logger.info(f"Saved final model to {config.MODEL_PATH}")

    logger.info("Evaluating on test set...")
    evaluate_model(model, data["X_test"], data["y_test"], data["label_encoder"])


if __name__ == "__main__":
    try:
        main()
    except cleaning.DatasetNotFoundError as e:
        print(str(e))
        sys.exit(1)
