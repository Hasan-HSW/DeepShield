"""
predict.py
----------
Inference-time module: loads the trained model + saved scaler/encoder,
preprocesses a new CSV file the same way training data was processed, and
returns the predicted attack class plus a risk score (0-100) derived from
the model's softmax confidence.

Usage as a script:
    python -m src.predict path/to/new_traffic.csv

Used as a library by app.py for the Flask /predict endpoint.
"""

import logging

import numpy as np
from tensorflow import keras

from src import config, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model = None  # lazy-loaded singleton so Flask doesn't reload it per request


def get_model():
    """Loads the trained model once and reuses it across calls."""
    global _model
    if _model is None:
        logger.info(f"Loading model from {config.MODEL_PATH}")
        _model = keras.models.load_model(config.MODEL_PATH)
    return _model


def compute_risk_score(confidence: float, predicted_class: str) -> float:
    """
    Maps the model's softmax confidence to a 0-100 risk score.

    "Normal" traffic gets an inverted score (high model confidence that
    traffic is Normal = LOW risk), while every attack class gets a risk
    score that scales directly with prediction confidence.
    """
    if predicted_class == "Normal":
        return round((1 - confidence) * 100, 2)
    return round(confidence * 100, 2)


def predict_from_csv(csv_path: str) -> dict:
    """
    Runs the full inference pipeline on a single uploaded CSV file.

    Returns:
        dict with predicted_class, confidence, risk_score, and
        should_alert (True if risk_score exceeds the configured threshold).
    """
    model = get_model()

    X_seq = pipeline.run_inference_pipeline(csv_path)
    probs = model.predict(X_seq)[0]

    from src.encoding import load_label_encoder
    label_encoder = load_label_encoder()

    predicted_index = int(np.argmax(probs))
    predicted_class = label_encoder.classes_[predicted_index]
    confidence = float(probs[predicted_index])

    risk_score = compute_risk_score(confidence, predicted_class)

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "risk_score": risk_score,
        "should_alert": risk_score > config.RISK_ALERT_THRESHOLD,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <path_to_csv>")
        sys.exit(1)

    result = predict_from_csv(sys.argv[1])
    print(result)
