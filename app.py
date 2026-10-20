"""
app.py
------
Full Flask web application for the DeepShield Intrusion Detection project.

Provides:
    GET  /            -> Home page (upload form)
    POST /analyze      -> Handles the CSV upload from the web UI, runs the
                          LSTM pipeline, and renders a results page
    POST /predict       -> JSON API (unchanged behaviour), for programmatic use
    GET  /health        -> Liveness check

Run from the project root:
    python app.py
Then open:
    http://localhost:5000
"""

import logging
import os
import tempfile
import traceback

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from src import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# When True, unexpected errors return their real message in the JSON/HTML
# response, which is useful while testing locally. Set to False before any
# real deployment so internal details aren't exposed publicly.
DEBUG_ERRORS = True

ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload limit


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def risk_level(risk_score: float) -> str:
    """Maps a 0-100 risk score to a color band used by the UI."""
    if risk_score >= 70:
        return "red"
    if risk_score >= 40:
        return "yellow"
    return "green"


def run_prediction(tmp_path: str):
    """
    Shared inference call used by both the web UI (/analyze) and the JSON
    API (/predict). Returns (result_dict, None) on success, or
    (None, (error_message, http_status)) on failure.
    """
    try:
        result = predict.predict_from_csv(tmp_path)
        return result, None

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return None, (str(e), 400)

    except KeyError as e:
        logger.warning(f"Schema error: {e}")
        return None, (f"Invalid file schema: {e}", 400)

    except (FileNotFoundError, OSError) as e:
        logger.error(f"Missing model artifact: {e}")
        return None, (
            "Model artifacts not found. Run 'python -m src.train' before "
            "starting the app.", 500
        )

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Unexpected error:\n{tb}")

        msg = str(e)
        if "deserializ" in msg.lower() or "GlorotUniform" in msg or "get_config" in msg:
            return None, (
                "تعذر تحميل النموذج بسبب اختلاف إصدار TensorFlow/Keras عن "
                "الإصدار الذي حُفظ به النموذج الحالي. الحل: احذف الملفات "
                "الموجودة داخل مجلد models/ ثم شغّل الأمر التالي مرة واحدة "
                "لتدريب نموذج مطابق تماماً لإصدارك المثبت: "
                "python -m src.train", 500
            )
        if DEBUG_ERRORS:
            return None, (f"Internal server error: {type(e).__name__}: {e}", 500)
        return None, ("Internal server error while processing the file.", 500)


# --------------------------------------------------------------------------
# Web UI routes
# --------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template(
            "index.html",
            error="لم يتم اختيار أي ملف. الرجاء اختيار ملف CSV أولاً.",
        )

    file = request.files["file"]

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            error="نوع الملف غير مدعوم. الرجاء رفع ملف بصيغة CSV فقط (.csv).",
        )

    filename = secure_filename(file.filename)
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, filename)
    file.save(tmp_path)

    try:
        result, error = run_prediction(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        os.rmdir(tmp_dir)

    if error:
        message, _status = error
        return render_template("index.html", error=message)

    result["risk_level"] = risk_level(result["risk_score"])
    result["confidence_percent"] = round(result["confidence"] * 100, 2)

    return render_template("result.html", filename=filename, **result)


# --------------------------------------------------------------------------
# JSON API (kept for programmatic / automated use)
# --------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Accepts an uploaded CSV of network traffic flows and returns the
    predicted attack classification as JSON.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request. Expected field 'file'."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only .csv files are supported."}), 400

    filename = secure_filename(file.filename)
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, filename)
    file.save(tmp_path)

    try:
        result, error = run_prediction(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        os.rmdir(tmp_dir)

    if error:
        message, status = error
        return jsonify({"error": message}), status

    logger.info(f"Prediction for {filename}: {result}")
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
