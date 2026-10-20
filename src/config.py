"""
config.py
---------
Central place for paths, constants, and hyperparameters used across the
whole pipeline (preprocessing, model, training, inference, API). Every
other module imports from here instead of hardcoding values, so the
project is easy to tune from one place.
"""

import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(MODELS_DIR, "model.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.joblib")

CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
METRICS_REPORT_PATH = os.path.join(OUTPUTS_DIR, "metrics_report.txt")

# --------------------------------------------------------------------------
# Real CIC-IDS-2017 dataset — official source only
# --------------------------------------------------------------------------
# This project trains ONLY on the real CIC-IDS-2017 dataset published by the
# Canadian Institute for Cybersecurity (CIC), University of New Brunswick.
# No sample/synthetic/random data is used anywhere in the training path.
DATASET_OFFICIAL_PAGE = "https://www.unb.ca/cic/datasets/ids-2017.html"
DATASET_DOWNLOAD_URL = (
    "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/"
    "CSVs/MachineLearningCSV.zip"
)
DATASET_DOWNLOAD_MD5_URL = (
    "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/"
    "CSVs/MachineLearningCSV.md5"
)

# Filename fragments (case-insensitive) that mark a CSV as NOT being real
# CICIDS2017 data (leftover smoke-test/sample/dummy files). Any such file
# found in DATA_DIR is skipped with a warning instead of being trained on.
FORBIDDEN_FILENAME_MARKERS = ["sample", "synthetic", "dummy", "fake", "random"]

# The 8 official CICIDS2017 daily-capture CSV files (MachineLearningCSV.zip).
# Used only to print a helpful "X/8 official files found" hint — the loader
# still accepts any real CICIDS2017 CSV placed in DATA_DIR, it does not
# hard-require these exact filenames.
OFFICIAL_DATASET_FILENAMES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# A real CICIDS2017 daily CSV has tens of thousands of rows at minimum.
# Anything smaller in DATA_DIR is almost certainly not the real dataset
# and is rejected rather than silently trained on.
MIN_ROWS_PER_FILE = 1000

# --------------------------------------------------------------------------
# Label mapping — CICIDS2017's raw labels -> DeepShield's 6 target classes
# --------------------------------------------------------------------------
# CICIDS2017 CSVs use fine-grained attack names. Anything not explicitly
# listed here falls into "Other" via the .get() default in cleaning.py's
# label mapping step, rather than being silently dropped — this keeps rare
# attack types (Heartbleed, Infiltration, etc.) in the dataset instead of
# discarding them, at the cost of that "Other" class being a mixed bag.
LABEL_MAP = {
    "BENIGN": "Normal",

    "DDoS": "DDoS",
    "DoS Hulk": "DDoS",
    "DoS GoldenEye": "DDoS",
    "DoS slowloris": "DDoS",
    "DoS Slowhttptest": "DDoS",

    "PortScan": "PortScan",

    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",

    # The official CSVs encode the separator between "Web Attack" and the
    # sub-type (Brute Force / XSS / Sql Injection) as a mojibake byte that
    # is NOT recoverable losslessly (it decodes to U+FFFD, the Unicode
    # replacement character, no matter which encoding you open the file
    # with — the original byte was already lost when CIC generated these
    # CSVs). An exact-string dict key can't reliably match that, so these
    # three entries are kept only for documentation; the actual matching
    # happens by prefix in encoding.map_labels() ("Web Attack" -> "Web
    # Attack", regardless of the mangled separator/suffix).
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",

    "Bot": "Other",
    "Infiltration": "Other",
    "Heartbleed": "Other",
}
DEFAULT_LABEL = "Other"  # fallback for any raw label not in LABEL_MAP above

TARGET_CLASSES = ["Normal", "DDoS", "PortScan", "Brute Force", "Web Attack", "Other"]

# --------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------
# Column names must match the CICIDS2017 CSV headers exactly (after
# whitespace stripping — see cleaning.strip_column_names).
#
# NOTE: the official MachineLearningCSV.zip export of CICIDS2017 (the one
# downloaded from the CIC/UNB site, and produced by download_dataset.py)
# does NOT contain "Source Port" or "Protocol" columns — only 78 flow
# features + "Label". Earlier versions of this config referenced those two
# columns, which do not exist in the real dataset and made
# encoding.select_features() raise KeyError on every real CSV. They have
# been removed so the pipeline runs against the actual official files.
# If you later train on a CICIDS2017 export that DOES include them, add
# "Source Port" back to NUMERIC_FEATURES and "Protocol" to
# CATEGORICAL_FEATURES.
NUMERIC_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "SYN Flag Count",
    "ACK Flag Count",
    "Destination Port",
]

# No categorical columns in the official export. Kept as an empty list
# (rather than removed) so encoding.py's pd.get_dummies(columns=[...]) call
# keeps working unchanged if a categorical feature is added back later.
CATEGORICAL_FEATURES = []

SELECTED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LABEL_COLUMN = "Label"

# --------------------------------------------------------------------------
# Sequence construction
# --------------------------------------------------------------------------
TIMESTEPS = 10
SEQUENCE_STRIDE = 1

# --------------------------------------------------------------------------
# Train / val / test split
# --------------------------------------------------------------------------
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Training hyperparameters
# --------------------------------------------------------------------------
LSTM_UNITS = 64
DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
EPOCHS = 50

# --------------------------------------------------------------------------
# Risk scoring / alerting (used by app.py)
# --------------------------------------------------------------------------
RISK_ALERT_THRESHOLD = 90  # send alert if risk score (0-100) exceeds this
