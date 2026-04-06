# inference.py
# ─────────────────────────────────────────────────────────────────────────────
# Real-time / streaming fraud detection engine.
# Simulates a transaction stream and scores each transaction as it arrives.
# ─────────────────────────────────────────────────────────────────────────────

import time
import logging
import json
import queue
import threading
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Generator
from dataclasses import dataclass, asdict

from config import (
    MODEL_DIR, INFERENCE_CONFIG, FEATURE_CONFIG, SEED
)
from data_loader import FEATURE_COLS

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Transaction dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Transaction:
    step            : int
    tx_type         : str
    amount          : float
    name_orig       : str
    old_balance_orig: float
    new_balance_orig: float
    name_dest       : str
    old_balance_dest: float
    new_balance_dest: float

@dataclass
class FraudAlert:
    transaction     : Transaction
    fraud_proba     : float
    is_fraud        : bool
    risk_level      : str          # LOW / MEDIUM / HIGH / CRITICAL
    alert_reason    : str
    latency_ms      : float


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Feature extractor (single transaction → feature vector)
# ─────────────────────────────────────────────────────────────────────────────

TYPE_ENCODING = {
    "CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2,
    "PAYMENT": 3, "TRANSFER": 4,
}

def extract_features_single(tx: Transaction,
                              history: pd.DataFrame,
                              scaler) -> np.ndarray:
    """
    Extracts the same feature set used during training for a single transaction.

    Parameters
    ----------
    tx      : incoming Transaction
    history : recent transaction history (rolling window)
    scaler  : StandardScaler fitted during training
    """
    amount      = tx.amount
    amount_log  = np.log1p(amount)
    amount_zscore = 0.0   # would need per-type mean/std at runtime

    balance_diff_orig = tx.new_balance_orig - tx.old_balance_orig
    balance_diff_dest = tx.new_balance_dest - tx.old_balance_dest
    balance_ratio_orig = tx.new_balance_orig / (tx.old_balance_orig + 1)
    balance_ratio_dest = tx.new_balance_dest / (tx.old_balance_dest + 1)

    orig_balance_error = abs(
        max(tx.old_balance_orig - amount, 0) - tx.new_balance_orig
    )
    dest_balance_error = abs(
        tx.old_balance_dest + amount - tx.new_balance_dest
    )

    # Rolling aggregate over history for this account
    acct_hist = history[history["nameOrig"] == tx.name_orig] \
        if history is not None and len(history) > 0 else pd.DataFrame()

    orig_tx_count    = len(acct_hist) + 1
    orig_total_sent  = acct_hist["amount"].sum() + amount if len(acct_hist) > 0 else amount
    orig_unique_dests = acct_hist["nameDest"].nunique() if len(acct_hist) > 0 else 1

    feat = {
        "amount"            : amount,
        "amount_log"        : amount_log,
        "amount_zscore"     : amount_zscore,
        "is_round_amount"   : int(amount % 100 == 0),
        "oldbalanceOrg"     : tx.old_balance_orig,
        "newbalanceOrig"    : tx.new_balance_orig,
        "oldbalanceDest"    : tx.old_balance_dest,
        "newbalanceDest"    : tx.new_balance_dest,
        "balance_diff_orig" : balance_diff_orig,
        "balance_diff_dest" : balance_diff_dest,
        "balance_ratio_orig": balance_ratio_orig,
        "balance_ratio_dest": balance_ratio_dest,
        "orig_balance_error": orig_balance_error,
        "dest_balance_error": dest_balance_error,
        "has_orig_error"    : int(orig_balance_error > 0.01),
        "has_dest_error"    : int(dest_balance_error > 0.01),
        "orig_zero_before"  : int(tx.old_balance_orig == 0),
        "orig_zero_after"   : int(tx.new_balance_orig == 0),
        "dest_zero_before"  : int(tx.old_balance_dest == 0),
        "hour_of_day"       : tx.step % 24,
        "day_of_week"       : (tx.step // 24) % 7,
        "step_mod_24"       : tx.step % 24,
        "type_encoded"      : TYPE_ENCODING.get(tx.tx_type, 0),
        "orig_tx_count"     : orig_tx_count,
        "orig_total_sent"   : orig_total_sent,
        "orig_unique_dests" : orig_unique_dests,
    }

    vec = np.array([feat[col] for col in FEATURE_COLS], dtype=np.float32)
    vec = np.nan_to_num(vec)
    vec = scaler.transform(vec.reshape(1, -1))[0]
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Rule-Based Pre-filter (fast path before ML)
# ─────────────────────────────────────────────────────────────────────────────

def rule_based_score(tx: Transaction) -> float:
    """
    Quick heuristic rules that catch obvious fraud before invoking ML.
    Returns a score in [0, 1] — 0 = definitely safe, 1 = suspicious.
    """
    score = 0.0

    # Only CASH_OUT and TRANSFER are associated with fraud in PaySim
    if tx.tx_type in ("CASH_OUT", "TRANSFER"):
        score += 0.20

    # Full drain of origin account
    if tx.old_balance_orig > 0 and tx.new_balance_orig == 0:
        score += 0.30

    # Very large transaction
    if tx.amount > 500_000:
        score += 0.15

    # Destination balance unchanged (money laundering indicator)
    if tx.old_balance_dest == tx.new_balance_dest and tx.amount > 0:
        score += 0.25

    # Round amounts
    if tx.amount % 100_000 == 0 and tx.amount > 0:
        score += 0.10

    return min(score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Risk Level Classifier
# ─────────────────────────────────────────────────────────────────────────────

def classify_risk(proba: float) -> str:
    if proba >= 0.90: return "CRITICAL"
    if proba >= 0.70: return "HIGH"
    if proba >= 0.50: return "MEDIUM"
    return "LOW"

def build_alert_reason(tx: Transaction, proba: float, rule_score: float) -> str:
    reasons = []
    if tx.tx_type in ("CASH_OUT", "TRANSFER"):
        reasons.append(f"High-risk tx type ({tx.tx_type})")
    if tx.new_balance_orig == 0 and tx.old_balance_orig > 0:
        reasons.append("Origin account fully drained")
    if tx.old_balance_dest == tx.new_balance_dest:
        reasons.append("Destination balance unchanged (possible layering)")
    if tx.amount > 1_000_000:
        reasons.append(f"Very large amount (${tx.amount:,.0f})")
    if rule_score > 0.5:
        reasons.append(f"Rule engine score: {rule_score:.2f}")
    if proba > 0.7:
        reasons.append(f"ML model confidence: {proba:.2%}")
    return "; ".join(reasons) if reasons else "ML model flagged"


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FraudDetector  (main scoring class)
# ─────────────────────────────────────────────────────────────────────────────

class RealTimeFraudDetector:
    """
    Scores incoming transactions in real time.

    Uses a two-stage approach:
    1. Rule-based pre-filter (< 1 ms)
    2. ML ensemble (XGB + LGB)

    Maintains a rolling window of recent transactions per account
    for live feature computation.
    """

    def __init__(self,
                 ensemble_path: Path = None,
                 scaler_path:   Path = None,
                 window_hours:  int  = 24,
                 alert_threshold: float = None):

        ensemble_path   = ensemble_path or (MODEL_DIR / "ensemble.pkl")
        scaler_path     = scaler_path   or (MODEL_DIR / "scaler.pkl")
        self.alert_threshold = alert_threshold or INFERENCE_CONFIG["alert_threshold"]
        self.window_hours    = window_hours

        log.info(f"Loading ensemble from {ensemble_path} …")
        self.ensemble = joblib.load(ensemble_path)
        self.scaler   = joblib.load(scaler_path)

        # Rolling history: {account_id: list of recent transactions}
        self._history = pd.DataFrame()
        self._alert_count = 0
        self._total_count = 0

    def _update_history(self, tx: Transaction):
        """Append transaction to rolling window."""
        new_row = pd.DataFrame([{
            "nameOrig" : tx.name_orig,
            "nameDest" : tx.name_dest,
            "amount"   : tx.amount,
            "step"     : tx.step,
        }])
        self._history = pd.concat([self._history, new_row], ignore_index=True)

        # Keep only recent window
        max_step = self._history["step"].max()
        self._history = self._history[
            self._history["step"] >= max_step - self.window_hours
        ]

    def score(self, tx: Transaction) -> FraudAlert:
        """Score a single transaction. Returns a FraudAlert."""
        t0 = time.perf_counter()

        # Rule-based fast check
        rule_score = rule_based_score(tx)

        # Feature extraction
        feat_vec = extract_features_single(tx, self._history, self.scaler)

        # ML inference
        proba = float(self.ensemble.predict_proba(
            feat_vec.reshape(1, -1)
        )[0])

        # Blend rule + ML (rule gives uplift if suspicious)
        blended_proba = min(proba + rule_score * 0.1, 1.0)

        is_fraud   = blended_proba >= self.alert_threshold
        risk_level = classify_risk(blended_proba)
        reason     = build_alert_reason(tx, blended_proba, rule_score)
        latency_ms = (time.perf_counter() - t0) * 1000

        # Update rolling history
        self._update_history(tx)

        self._total_count += 1
        if is_fraud:
            self._alert_count += 1

        alert = FraudAlert(
            transaction = tx,
            fraud_proba = round(blended_proba, 4),
            is_fraud    = is_fraud,
            risk_level  = risk_level,
            alert_reason= reason,
            latency_ms  = round(latency_ms, 2),
        )
        return alert

    @property
    def stats(self):
        return {
            "total_scored" : self._total_count,
            "total_alerts" : self._alert_count,
            "alert_rate"   : self._alert_count / max(self._total_count, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Streaming Simulator
# ─────────────────────────────────────────────────────────────────────────────

def simulate_stream(detector: RealTimeFraudDetector,
                    df: pd.DataFrame,
                    delay_ms: float = None,
                    max_tx: int = 1000) -> Generator[FraudAlert, None, None]:
    """
    Simulates a real-time transaction stream from the test dataframe.
    Yields FraudAlert objects as transactions arrive.
    """
    delay = (delay_ms or INFERENCE_CONFIG["stream_delay_ms"]) / 1000.0
    df = df.head(max_tx).reset_index(drop=True)

    log.info(f"Starting stream simulation: {len(df)} transactions, "
             f"delay={delay*1000:.0f}ms …")

    for _, row in df.iterrows():
        tx = Transaction(
            step             = int(row.get("step", 0)),
            tx_type          = row.get("type", "TRANSFER"),
            amount           = float(row.get("amount", 0)),
            name_orig        = str(row.get("nameOrig", "C000")),
            old_balance_orig = float(row.get("oldbalanceOrg", 0)),
            new_balance_orig = float(row.get("newbalanceOrig", 0)),
            name_dest        = str(row.get("nameDest", "C001")),
            old_balance_dest = float(row.get("oldbalanceDest", 0)),
            new_balance_dest = float(row.get("newbalanceDest", 0)),
        )
        alert = detector.score(tx)
        yield alert
        time.sleep(delay)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Quick Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    detector = RealTimeFraudDetector(alert_threshold=0.60)

    # Load test data
    from config import SPLITS_PKL
    splits = joblib.load(SPLITS_PKL)
    test_df = splits["test_df"].head(200)

    alerts_raised = 0
    for alert in simulate_stream(detector, test_df, delay_ms=10, max_tx=200):
        status = "🚨 ALERT" if alert.is_fraud else "✅ OK"
        print(
            f"{status}  [{alert.risk_level}]  "
            f"${alert.transaction.amount:>12,.2f}  "
            f"proba={alert.fraud_proba:.3f}  "
            f"{alert.latency_ms:.1f}ms"
        )
        if alert.is_fraud:
            print(f"        → {alert.alert_reason}")
            alerts_raised += 1

    print(f"\nTotal: {detector.stats}")
