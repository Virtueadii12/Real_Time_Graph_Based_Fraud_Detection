# api.py
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI REST API exposing fraud detection endpoints.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import time
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import uvicorn

from config import API_CONFIG, MODEL_DIR
from inference import (
    RealTimeFraudDetector, Transaction, FraudAlert, simulate_stream
)

log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Graph-Based Fraud Detection API",
    description = "Real-time transaction fraud scoring using GNN + XGB + LGB ensemble",
    version     = "1.0.0",
    docs_url    = "/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Global detector (loaded once at startup) ──────────────────────────────────
detector: Optional[RealTimeFraudDetector] = None


@app.on_event("startup")
async def load_models():
    global detector
    log.info("Loading fraud detection models …")
    try:
        detector = RealTimeFraudDetector(alert_threshold=0.60)
        log.info("Models loaded successfully ✅")
    except FileNotFoundError:
        log.error("Model files not found. Run train.py first.")
        detector = None


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    step              : int   = Field(..., ge=0, description="Hour since simulation start")
    type              : str   = Field(..., description="CASH_IN | CASH_OUT | DEBIT | PAYMENT | TRANSFER")
    amount            : float = Field(..., gt=0)
    nameOrig          : str
    oldbalanceOrg     : float
    newbalanceOrig    : float
    nameDest          : str
    oldbalanceDest    : float
    newbalanceDest    : float

class BatchRequest(BaseModel):
    transactions: List[TransactionRequest]

class FraudScoreResponse(BaseModel):
    nameOrig          : str
    nameDest          : str
    amount            : float
    fraud_probability : float
    is_fraud          : bool
    risk_level        : str
    alert_reason      : str
    latency_ms        : float

class BatchResponse(BaseModel):
    results           : List[FraudScoreResponse]
    total_flagged     : int
    total_processed   : int
    processing_time_ms: float

class HealthResponse(BaseModel):
    status            : str
    model_loaded      : bool
    total_scored      : int
    alert_rate        : float


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Health check endpoint."""
    if detector is None:
        return HealthResponse(status="degraded", model_loaded=False,
                               total_scored=0, alert_rate=0.0)
    stats = detector.stats
    return HealthResponse(
        status       = "ok",
        model_loaded = True,
        total_scored = stats["total_scored"],
        alert_rate   = round(stats["alert_rate"], 4),
    )


@app.post("/score", response_model=FraudScoreResponse, tags=["Fraud Detection"])
async def score_transaction(request: TransactionRequest):
    """
    Score a single transaction in real time.

    Returns fraud probability, risk level, and alert reason.
    """
    if detector is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Run train.py first.")

    tx = Transaction(
        step             = request.step,
        tx_type          = request.type,
        amount           = request.amount,
        name_orig        = request.nameOrig,
        old_balance_orig = request.oldbalanceOrg,
        new_balance_orig = request.newbalanceOrig,
        name_dest        = request.nameDest,
        old_balance_dest = request.oldbalanceDest,
        new_balance_dest = request.newbalanceDest,
    )
    alert = detector.score(tx)

    return FraudScoreResponse(
        nameOrig          = tx.name_orig,
        nameDest          = tx.name_dest,
        amount            = tx.amount,
        fraud_probability = alert.fraud_proba,
        is_fraud          = alert.is_fraud,
        risk_level        = alert.risk_level,
        alert_reason      = alert.alert_reason,
        latency_ms        = alert.latency_ms,
    )


@app.post("/score/batch", response_model=BatchResponse, tags=["Fraud Detection"])
async def score_batch(request: BatchRequest):
    """
    Score a batch of transactions (up to 1000 per request).
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if len(request.transactions) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 transactions per batch.")

    t0      = time.perf_counter()
    results = []
    flagged = 0

    for req_tx in request.transactions:
        tx = Transaction(
            step             = req_tx.step,
            tx_type          = req_tx.type,
            amount           = req_tx.amount,
            name_orig        = req_tx.nameOrig,
            old_balance_orig = req_tx.oldbalanceOrg,
            new_balance_orig = req_tx.newbalanceOrig,
            name_dest        = req_tx.nameDest,
            old_balance_dest = req_tx.oldbalanceDest,
            new_balance_dest = req_tx.newbalanceDest,
        )
        alert = detector.score(tx)
        results.append(FraudScoreResponse(
            nameOrig          = tx.name_orig,
            nameDest          = tx.name_dest,
            amount            = tx.amount,
            fraud_probability = alert.fraud_proba,
            is_fraud          = alert.is_fraud,
            risk_level        = alert.risk_level,
            alert_reason      = alert.alert_reason,
            latency_ms        = alert.latency_ms,
        ))
        if alert.is_fraud:
            flagged += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return BatchResponse(
        results            = results,
        total_flagged      = flagged,
        total_processed    = len(results),
        processing_time_ms = round(elapsed_ms, 1),
    )


@app.get("/stats", tags=["System"])
async def get_stats():
    """Returns running statistics of the detector."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return detector.stats


@app.get("/example/fraud", tags=["Demo"])
async def example_fraud_transaction():
    """Returns an example fraud transaction score (TRANSFER that fully drains origin)."""
    sample = TransactionRequest(
        step          = 1,
        type          = "TRANSFER",
        amount        = 181_374.00,
        nameOrig      = "C1231006815",
        oldbalanceOrg = 181_374.00,
        newbalanceOrig= 0.00,
        nameDest      = "C553264065",
        oldbalanceDest= 0.00,
        newbalanceDest= 0.00,
    )
    return await score_transaction(sample)


@app.get("/example/legit", tags=["Demo"])
async def example_legit_transaction():
    """Returns an example legitimate transaction score."""
    sample = TransactionRequest(
        step          = 1,
        type          = "PAYMENT",
        amount        = 9839.64,
        nameOrig      = "C1231006815",
        oldbalanceOrg = 170_136.00,
        newbalanceOrig= 160_296.36,
        nameDest      = "M1979787155",
        oldbalanceDest= 0.00,
        newbalanceDest= 0.00,
    )
    return await score_transaction(sample)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    uvicorn.run(
        "api:app",
        host   = API_CONFIG["host"],
        port   = API_CONFIG["port"],
        reload = API_CONFIG["reload"],
    )
