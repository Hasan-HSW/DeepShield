"""
cleaning.py
-----------
Loads and cleans the REAL CIC-IDS-2017 dataset. This module refuses to
train on anything else: no sklearn sample datasets, no random/dummy data,
no placeholder CSVs. If a valid real dataset is not present in DATA_DIR,
every entry point here raises DatasetNotFoundError with instructions on
how to obtain it.

Two loading paths are supported:

- load_all_csvs_from_folder(): reads every real CICIDS2017 .csv in
  DATA_DIR, validates each one individually, aligns columns across all of
  them, and concatenates them into a single DataFrame (CICIDS2017 ships
  as one CSV per day-session capture).
- load_raw_csv(): reads a single CSV (used by predict.py for one uploaded
  file at inference time).

Cleaning handles the dataset's known issues: whitespace in column names,
Infinity values, NaNs, non-numeric junk in numeric columns, and duplicate
rows.
"""

import glob
import logging
import os

import numpy as np
import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetNotFoundError(FileNotFoundError):
    """
    Raised whenever DATA_DIR does not contain a usable, real CIC-IDS-2017
    dataset — missing folder, no CSVs, or every CSV present failed
    validation (empty / corrupted / no Label column / too small /
    flagged as sample-synthetic data).
    """
    pass


def _no_real_dataset_error(folder: str) -> DatasetNotFoundError:
    message = (
        "\n"
        "============================================================\n"
        "  NO REAL CIC-IDS-2017 DATA FOUND\n"
        "============================================================\n"
        f"  Expected location: {os.path.abspath(folder)}\n"
        "\n"
        "  This project trains ONLY on the real CIC-IDS-2017 dataset.\n"
        "  It will not run on sample, synthetic, random, or sklearn\n"
        "  toy data of any kind.\n"
        "\n"
        "  How to fix this:\n"
        f"    1. Official dataset page:\n"
        f"       {config.DATASET_OFFICIAL_PAGE}\n"
        f"    2. Direct download (MachineLearningCSV.zip, ~224 MB):\n"
        f"       {config.DATASET_DOWNLOAD_URL}\n"
        f"    3. Extract the .csv files and place them directly in:\n"
        f"       {os.path.abspath(folder)}\n"
        "\n"
        "  Or, on a machine with internet access, run from the project\n"
        "  root:\n"
        "       python download_dataset.py\n"
        "  which downloads and extracts the official files into data/\n"
        "  for you automatically.\n"
        "============================================================\n"
    )
    return DatasetNotFoundError(message)


def _is_forbidden_filename(filename: str) -> bool:
    """True if the filename suggests sample/synthetic/dummy data."""
    lower = filename.lower()
    return any(marker in lower for marker in config.FORBIDDEN_FILENAME_MARKERS)


def load_all_csvs_from_folder(folder: str = config.DATA_DIR) -> pd.DataFrame:
    """
    Loads every real CIC-IDS-2017 CSV file in `folder`, validating and
    reporting on each one, then merges them into a single DataFrame with
    consistent columns.

    Validation per file (any failure -> file is skipped, not fatal on its
    own):
        - filename doesn't look like sample/synthetic/dummy data
        - file parses as a valid CSV
        - file is not empty
        - file has a "Label" column (after stripping column whitespace)
        - file has at least config.MIN_ROWS_PER_FILE rows

    Raises:
        DatasetNotFoundError: if `folder` doesn't exist, contains no CSV
            files at all, or every CSV present fails validation.
    """
    if not os.path.isdir(folder):
        raise _no_real_dataset_error(folder)

    all_csv_paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not all_csv_paths:
        raise _no_real_dataset_error(folder)

    frames = []
    loaded_files = []
    skipped_files = []

    for path in all_csv_paths:
        name = os.path.basename(path)

        if _is_forbidden_filename(name):
            reason = "filename indicates sample/synthetic/dummy data, not real CICIDS2017 data"
            logger.warning(f"Skipping '{name}': {reason}")
            skipped_files.append((name, reason))
            continue

        try:
            df = pd.read_csv(path, low_memory=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError,
                UnicodeDecodeError, OSError) as e:
            reason = f"failed to parse ({type(e).__name__}: {e})"
            logger.warning(f"Skipping '{name}': {reason}")
            skipped_files.append((name, reason))
            continue

        if df is None or df.empty:
            reason = "file is empty"
            logger.warning(f"Skipping '{name}': {reason}")
            skipped_files.append((name, reason))
            continue

        df.columns = [c.strip() for c in df.columns]

        if config.LABEL_COLUMN not in df.columns:
            reason = f"no '{config.LABEL_COLUMN}' column found"
            logger.warning(f"Skipping '{name}': {reason}")
            skipped_files.append((name, reason))
            continue

        if len(df) < config.MIN_ROWS_PER_FILE:
            reason = f"only {len(df)} rows (< {config.MIN_ROWS_PER_FILE}); too small for a real daily capture"
            logger.warning(f"Skipping '{name}': {reason}")
            skipped_files.append((name, reason))
            continue

        logger.info(f"Loaded '{name}': {df.shape[0]} rows, {df.shape[1]} columns")
        frames.append(df)
        loaded_files.append(name)

    # ---- Reporting (required by the data-validation step) ----
    print(f"\nCSV files found in {os.path.abspath(folder)}: {len(all_csv_paths)}")
    print(f"Successfully loaded: {len(loaded_files)}")
    for n in loaded_files:
        print(f"   OK   {n}")
    if skipped_files:
        print(f"Skipped: {len(skipped_files)}")
        for n, reason in skipped_files:
            print(f"   SKIP {n} -> {reason}")

    official_found = sum(1 for n in loaded_files if n in config.OFFICIAL_DATASET_FILENAMES)
    print(f"Official CICIDS2017 daily files matched by name: {official_found}/{len(config.OFFICIAL_DATASET_FILENAMES)}")

    if not frames:
        raise _no_real_dataset_error(folder)

    # ---- Ensure consistent columns across all files before merging ----
    # Some CICIDS2017 exports differ slightly between sources (extra
    # trailing columns, column order). Keep only columns common to EVERY
    # loaded file, in the column order of the first file, so pd.concat
    # produces one clean, consistent schema.
    common_columns = set(frames[0].columns)
    for df in frames[1:]:
        common_columns &= set(df.columns)

    if config.LABEL_COLUMN not in common_columns:
        raise DatasetNotFoundError(
            f"The loaded CSV files do not share a common '{config.LABEL_COLUMN}' "
            f"column, so they cannot be safely merged. Check that every file in "
            f"{os.path.abspath(folder)} genuinely belongs to the CIC-IDS-2017 "
            f"dataset."
        )

    ordered_common_columns = [c for c in frames[0].columns if c in common_columns]
    frames = [df[ordered_common_columns] for df in frames]

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined dataset shape: {combined.shape}")
    print(f"\nDataset path in use: {os.path.abspath(folder)}")
    print(f"Total rows after merging {len(frames)} file(s): {len(combined):,}")
    print(f"Total columns (before cleaning): {combined.shape[1]}")

    return combined


def load_raw_csv(path: str) -> pd.DataFrame:
    """Loads a single CSV file (used for one uploaded file at inference time)."""
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {path} with shape {df.shape}")
    return df


def strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Removes leading/trailing whitespace from all column names."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def convert_features_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forces every configured numeric feature column to a numeric dtype.
    Any stray non-numeric value (corrupted cell, stray text) becomes NaN,
    which drop_missing() then removes — this is what guarantees the model
    only ever sees real numeric data.
    """
    df = df.copy()
    for col in config.NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def replace_inf_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Converts +/- Infinity to NaN so a single missing-value strategy handles both."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drops any row containing at least one NaN value."""
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if before > 0:
        logger.info(f"Dropped {dropped} rows containing NaN/Inf ({dropped / before:.2%})")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Dropped {before - len(df)} duplicate rows")
    return df


def validate_not_empty(df: pd.DataFrame) -> None:
    """
    Raises ValueError if the DataFrame is empty. This is what produces the
    "Empty CSV" / "Invalid data" errors in the V&V test suite and the
    Flask /predict route's error response.
    """
    if df is None or df.empty:
        raise ValueError("Dataset contains no valid rows after cleaning.")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full cleaning sequence on an already-loaded DataFrame."""
    df = strip_column_names(df)
    df = convert_features_to_numeric(df)
    df = replace_inf_with_nan(df)
    df = drop_missing(df)
    df = drop_duplicates(df)
    validate_not_empty(df)
    return df


def clean_pipeline_from_folder(folder: str = config.DATA_DIR) -> pd.DataFrame:
    """Loads every real CSV in `folder` and returns a single cleaned DataFrame."""
    df = load_all_csvs_from_folder(folder)
    return clean_dataframe(df)


def clean_pipeline_single_file(path: str) -> pd.DataFrame:
    """Loads and cleans one CSV file (used at inference time)."""
    df = load_raw_csv(path)
    return clean_dataframe(df)
