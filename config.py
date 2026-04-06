# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration for the Graph-Based Fraud Detection System
# ─────────────────────────────────────────────────────────────────────────────

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models" / "saved"
LOG_DIR    = BASE_DIR / "logs"
REPORT_DIR = BASE_DIR / "reports"

for d in [DATA_DIR, MODEL_DIR, LOG_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_NAME   = "ntnu-eksamen/paysim1"   # Kaggle dataset slug
RAW_CSV        = DATA_DIR / "PS_20174392719_1491204439457_log.csv"
PROCESSED_CSV  = DATA_DIR / "processed_transactions.csv"
GRAPH_PKL      = DATA_DIR / "transaction_graph.pkl"
SPLITS_PKL     = DATA_DIR / "data_splits.pkl"

# ── Graph Construction ────────────────────────────────────────────────────────
GRAPH_CONFIG = {
    "node_types"        : ["account"],           # homogeneous graph
    "edge_weight_col"   : "amount",
    "min_tx_amount"     : 0.0,
    "max_graph_nodes"   : 200_000,               # cap for memory
    "self_loops"        : False,
}

# ── Feature Engineering ───────────────────────────────────────────────────────
FEATURE_CONFIG = {
    "temporal_windows"  : [1, 6, 24],            # hours
    "graph_features"    : [
        "degree", "in_degree", "out_degree",
        "pagerank", "betweenness_centrality",
        "clustering_coeff", "avg_neighbor_degree",
    ],
    "transaction_features": [
        "amount", "amount_log", "amount_zscore",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "balance_diff_orig", "balance_diff_dest",
        "balance_ratio_orig", "balance_ratio_dest",
        "is_round_amount", "hour_of_day", "step_mod_24",
    ],
    "categorical_cols"  : ["type"],
    "target_col"        : "isFraud",
}

# ── GNN Model ─────────────────────────────────────────────────────────────────
GNN_CONFIG = {
    "model_type"        : "GraphSAGE",           # GraphSAGE | GAT | GCN
    "hidden_channels"   : 128,
    "num_layers"        : 3,
    "dropout"           : 0.3,
    "heads"             : 4,                     # for GAT only
    "aggregation"       : "mean",                # mean | max | lstm
    "lr"                : 1e-3,
    "weight_decay"      : 1e-4,
    "epochs"            : 100,
    "patience"          : 15,
    "batch_size"        : 4096,
    "neg_sampling_ratio": 3.0,
}

# ── Ensemble / XGBoost ────────────────────────────────────────────────────────
XGB_CONFIG = {
    "n_estimators"      : 500,
    "max_depth"         : 7,
    "learning_rate"     : 0.05,
    "subsample"         : 0.8,
    "colsample_bytree"  : 0.8,
    "scale_pos_weight"  : 100,                   # class imbalance
    "eval_metric"       : "aucpr",
    "early_stopping_rounds": 30,
    "tree_method"       : "hist",
    "device"            : "cpu",
}

LGB_CONFIG = {
    "n_estimators"      : 500,
    "max_depth"         : 7,
    "learning_rate"     : 0.05,
    "num_leaves"        : 63,
    "subsample"         : 0.8,
    "colsample_bytree"  : 0.8,
    "class_weight"      : "balanced",
    "metric"            : "average_precision",
    "early_stopping_rounds": 30,
}

# ── Ensemble Stacking ─────────────────────────────────────────────────────────
ENSEMBLE_CONFIG = {
    "weights"           : {"xgb": 0.35, "lgb": 0.35, "gnn": 0.30},
    "threshold"         : 0.50,                  # classification threshold
    "calibrate"         : True,
}

# ── Inference / Streaming ─────────────────────────────────────────────────────
INFERENCE_CONFIG = {
    "batch_size"        : 256,
    "alert_threshold"   : 0.70,
    "stream_delay_ms"   : 50,                    # simulated stream interval
}

# ── API ───────────────────────────────────────────────────────────────────────
API_CONFIG = {
    "host"              : "0.0.0.0",
    "port"              : 8000,
    "reload"            : False,
}

# ── Random Seed ───────────────────────────────────────────────────────────────
SEED = 42
