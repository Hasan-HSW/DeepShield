"""
download_dataset.py
--------------------
Downloads the OFFICIAL real CIC-IDS-2017 dataset, published by the
Canadian Institute for Cybersecurity (CIC), University of New Brunswick,
and extracts the CSV files directly into this project's data/ folder.

IMPORTANT: This must be run on a machine with an active internet
connection that can reach 205.174.165.80 (the CIC's own file server).
It cannot run inside a network-restricted sandbox.

Usage (from the project root):
    python download_dataset.py

Source / citation:
    Sharafaldin, I., Lashkari, A.H., Ghorbani, A.A. (2018),
    "Toward Generating a New Intrusion Detection Dataset and Intrusion
    Traffic Characterization", ICISSP 2018.
    https://www.unb.ca/cic/datasets/ids-2017.html
"""

import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DATASET_URL = (
    "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/"
    "CSVs/MachineLearningCSV.zip"
)
MD5_URL = (
    "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/"
    "CSVs/MachineLearningCSV.md5"
)
ZIP_FILENAME = "MachineLearningCSV.zip"


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded * 100 / total_size, 100)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(
            f"\r  Downloading... {pct:5.1f}%  ({mb_done:,.1f} / {mb_total:,.1f} MB)"
        )
        sys.stdout.flush()


def download_file(url: str, dest_path: str) -> None:
    print(f"Downloading:\n  {url}\n  -> {dest_path}")
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=_progress_hook)
        print()
    except Exception as e:
        raise RuntimeError(
            f"Failed to download the dataset from {url}\n"
            f"Reason: {e}\n\n"
            f"If this keeps failing (corporate firewall, proxy, etc.), "
            f"download the file manually from that URL in your browser "
            f"and place the extracted .csv files directly into:\n"
            f"  {DATA_DIR}"
        ) from e


def try_verify_md5(zip_path: str, md5_url: str) -> None:
    """Best-effort MD5 check. A failure here never blocks extraction."""
    try:
        with urllib.request.urlopen(md5_url, timeout=15) as resp:
            expected = resp.read().decode("utf-8", errors="ignore").strip().split()[0].lower()
    except Exception as e:
        print(f"  (Skipping MD5 verification -- could not fetch checksum: {e})")
        return

    print("  Verifying MD5 checksum...")
    hasher = hashlib.md5()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest().lower()

    if actual == expected:
        print("  MD5 OK -- file integrity verified.")
    else:
        print(
            f"  WARNING: MD5 mismatch (expected {expected}, got {actual}). "
            f"The download may be corrupted. Consider deleting "
            f"{zip_path} and running this script again."
        )


def extract_csvs(zip_path: str, target_dir: str) -> list:
    print(f"Extracting CSV files into: {target_dir}")
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".csv"):
                continue
            filename = os.path.basename(member)
            if not filename:
                continue
            out_path = os.path.join(target_dir, filename)
            with zf.open(member) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(filename)
    return extracted


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    placeholder = os.path.join(DATA_DIR, "PUT_REAL_CICIDS2017_CSVS_HERE.txt")
    if os.path.exists(placeholder):
        os.remove(placeholder)

    existing_csvs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    if existing_csvs:
        print(f"data/ already contains {len(existing_csvs)} CSV file(s):")
        for f in existing_csvs:
            print(f"  - {f}")
        answer = input("Download and add the official dataset anyway? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted. Existing files were left untouched.")
            return

    zip_path = os.path.join(DATA_DIR, ZIP_FILENAME)

    download_file(DATASET_URL, zip_path)
    try_verify_md5(zip_path, MD5_URL)
    extracted = extract_csvs(zip_path, DATA_DIR)
    os.remove(zip_path)

    if not extracted:
        raise RuntimeError(
            "The archive downloaded but no .csv files were found inside it. "
            "The official archive layout may have changed -- please check "
            "https://www.unb.ca/cic/datasets/ids-2017.html manually."
        )

    print(f"\nDone. Extracted {len(extracted)} CSV file(s) into {DATA_DIR}:")
    for f in extracted:
        print(f"  - {f}")
    print("\nYou can now train the model with:\n  python -m src.train")


if __name__ == "__main__":
    main()
