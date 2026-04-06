# data_loader.py
# ─────────────────────────────────────────────────────────────────────────────
# Downloads PaySim dataset from Kaggle, validates, and preprocesses it.
# ─────────────────────────────────────────────────────────────────────────────

import os
import logging
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

from config import (
    DATA_DIR, MODEL_DIR, DATASET_NAME,
    RAW_CSV, PROCESSED_CSV, SPLITS_PKL,
    FEATURE_CONFIG, SEED
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Download
# ─────────────────────────────────────────────────────────────────────────────

def download_dataset() -> Path:
    """
    Downloads the PaySim dataset from Kaggle using the Kaggle API.

    Prerequisites
    -------------
    1. pip install kaggle
    2. Place kaggle.json in ~/.kaggle/   (get it from kaggle.com → Account → API)
    3. chmod 600 ~/.kaggle/kaggle.json

    Returns
    -------
    Path to the raw CSV file.
    """
    if RAW_CSV.exists():
        log.info(f"Dataset already present at {RAW_CSV}")
        return RAW_CSV

    log.info(f"Downloading dataset '{DATASET_NAME}' from Kaggle …")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", DATASET_NAME,
        "-p", str(DATA_DIR),
        "--unzip"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed:\n{result.stderr}\n\n"
            "Make sure your kaggle.json API key is configured correctly.\n"
            "See: https://github.com/Kaggle/kaggle-api#api-credentials"
        )

    # The file may be named differently on some Kaggle mirrors
    candidates = list(DATA_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError("No CSV found after download. Check the dataset name.")

    csv_path = candidates[0]
    log.info(f"Downloaded → {csv_path}")
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load & Validate
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "step", "type", "amount",
    "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud",
]

def load_raw(path: Path = None) -> pd.DataFrame:
    """Loads raw PaySim CSV and validates schema."""
    path = path or RAW_CSV
    log.info(f"Loading raw data from {path} …")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"  Shape: {df.shape}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    log.info(f"  Fraud rate: {df['isFraud'].mean()*100:.4f}%")
    log.info(f"  Transaction types:\n{df['type'].value_counts()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates rich feature set from raw PaySim data.

    Features Added
    --------------
    - Log-transformed amount
    - Z-score of amount per transaction type
    - Balance differentials and ratios
    - Round-amount flag
    - Hour of day, step mod 24
    - Error flags (balance mismatches)
    """
    log.info("Engineering features …")
    df = df.copy()

    # ── Amount features ──────────────────────────────────────────────────────
    df["amount_log"]   = np.log1p(df["amount"])
    df["amount_zscore"] = df.groupby("type")["amount"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    df["is_round_amount"] = (df["amount"] % 100 == 0).astype(int)

    # ── Balance differentials ────────────────────────────────────────────────
    df["balance_diff_orig"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["balance_ratio_orig"] = df["newbalanceOrig"] / (df["oldbalanceOrg"] + 1)
    df["balance_ratio_dest"] = df["newbalanceDest"] / (df["oldbalanceDest"] + 1)

    # ── Error flags (common fraud tell) ─────────────────────────────────────
    # Fraud transactions often don't correctly reduce origin balance
    df["orig_balance_error"] = (
        (df["oldbalanceOrg"] - df["amount"]).clip(lower=0) - df["newbalanceOrig"]
    ).abs()
    df["dest_balance_error"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    ).abs()
    df["has_orig_error"] = (df["orig_balance_error"] > 0.01).astype(int)
    df["has_dest_error"] = (df["dest_balance_error"] > 0.01).astype(int)

    # ── Zero-balance flags ───────────────────────────────────────────────────
    df["orig_zero_before"] = (df["oldbalanceOrg"] == 0).astype(int)
    df["orig_zero_after"]  = (df["newbalanceOrig"] == 0).astype(int)
    df["dest_zero_before"] = (df["oldbalanceDest"] == 0).astype(int)

    # ── Temporal features ─────────────────────────────────────────────────────
    df["hour_of_day"] = df["step"] % 24
    df["day_of_week"] = (df["step"] // 24) % 7
    df["step_mod_24"] = df["step"] % 24

    # ── Encode transaction type ───────────────────────────────────────────────
    le = LabelEncoder()
    df["type_encoded"] = le.fit_transform(df["type"])

    # ── Per-account aggregate stats ──────────────────────────────────────────
    # Number of transactions per origin account
    orig_counts = df.groupby("nameOrig")["isFraud"].transform("count")
    df["orig_tx_count"] = orig_counts

    # Total amount sent by origin
    df["orig_total_sent"] = df.groupby("nameOrig")["amount"].transform("sum")

    # Number of unique destinations
    df["orig_unique_dests"] = df.groupby("nameOrig")["nameDest"].transform("nunique")

    log.info(f"  Feature matrix shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Train / Val / Test Split
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "amount", "amount_log", "amount_zscore", "is_round_amount",
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "balance_diff_orig", "balance_diff_dest",
    "balance_ratio_orig", "balance_ratio_dest",
    "orig_balance_error", "dest_balance_error",
    "has_orig_error", "has_dest_error",
    "orig_zero_before", "orig_zero_after", "dest_zero_before",
    "hour_of_day", "day_of_week", "step_mod_24",
    "type_encoded",
    "orig_tx_count", "orig_total_sent", "orig_unique_dests",
]

def split_data(df: pd.DataFrame):
    """
    Temporal train/val/test split (preserves time ordering).

    PaySim has 744 steps (≈ 1 month).
    Train : steps 1–600
    Val   : steps 601–672
    Test  : steps 673–744
    """
    log.info("Splitting dataset …")
    target = FEATURE_CONFIG["target_col"]

    train = df[df["step"] <= 600].reset_index(drop=True)
    val   = df[(df["step"] > 600) & (df["step"] <= 672)].reset_index(drop=True)
    test  = df[df["step"] > 672].reset_index(drop=True)

    log.info(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    for name, subset in [("Train", train), ("Val", val), ("Test", test)]:
        log.info(f"  {name} fraud rate: {subset[target].mean()*100:.4f}%")

    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS].fillna(0))
    X_val   = scaler.transform(val[FEATURE_COLS].fillna(0))
    X_test  = scaler.transform(test[FEATURE_COLS].fillna(0))

    y_train = train[target].values
    y_val   = val[target].values
    y_test  = test[target].values

    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val"  : X_val,   "y_val"  : y_val,
        "X_test" : X_test,  "y_test" : y_test,
        "feature_cols": FEATURE_COLS,
        "train_df": train,
        "val_df"  : val,
        "test_df" : test,
    }

    joblib.dump(splits, SPLITS_PKL)
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    log.info(f"Splits saved → {SPLITS_PKL}")
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_data_pipeline(csv_path: Path = None) -> dict:
    """
    Full data pipeline:
      download → load → engineer features → split → save

    Parameters
    ----------
    csv_path : optional override (e.g. for unit tests)

    Returns
    -------
    dict with train/val/test arrays
    """
    if csv_path is None:
        csv_path = download_dataset()

    df_raw  = load_raw(csv_path)
    df_feat = engineer_features(df_raw)

    # Cache processed CSV
    df_feat.to_csv(PROCESSED_CSV, index=False)
    log.info(f"Processed CSV saved → {PROCESSED_CSV}")

    splits = split_data(df_feat)
    return splits


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    splits = run_data_pipeline()
    print(f"\nX_train: {splits['X_train'].shape}")
    print(f"X_val  : {splits['X_val'].shape}")
    print(f"X_test : {splits['X_test'].shape}")
    print(f"Fraud positives in test: {splits['y_test'].sum()}")
