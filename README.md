# 🔍 Real-Time Graph-Based Fraud Detection

> A production-grade fraud detection system combining **Graph Neural Networks**, **XGBoost**, and **LightGBM** on the PaySim financial transaction dataset.

---

## 📋 Table of Contents
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)

---

## 🏗️ Architecture

```
Raw PaySim CSV
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                   DATA PIPELINE                          │
│  Download → Validate → Feature Engineering → Split      │
└─────────────────────────┬───────────────────────────────┘
                           │
           ┌───────────────┼────────────────┐
           ▼               ▼                ▼
    ┌──────────┐   ┌──────────────┐  ┌──────────┐
    │ TABULAR  │   │    GRAPH     │  │   GRAPH  │
    │ FEATURES │   │ CONSTRUCTION │  │ FEATURES │
    │ (26 cols)│   │  (NetworkX)  │  │(PageRank,│
    └────┬─────┘   └──────┬───────┘  │ Betw., ..)
         │                │          └────┬──────┘
         │                ▼               │
         │         ┌──────────────┐       │
         │         │  GraphSAGE   │◄──────┘
         │         │  (3 layers)  │
         │         └──────┬───────┘
         │                │
         ▼                ▼
    ┌─────────┐    ┌──────────────┐
    │XGBoost  │    │  GNN Proba   │
    │LightGBM │    │  per node    │
    └────┬────┘    └──────┬───────┘
         │                │
         └────────┬────────┘
                  ▼
         ┌────────────────┐
         │ STACKED        │
         │ ENSEMBLE       │
         │ (Weighted Avg  │
         │  + Meta-LR)    │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │ FRAUD SCORE    │
         │ + RISK LEVEL   │
         │ + ALERT REASON │
         └────────────────┘
```

### Two-Stage Detection
1. **Rule Engine** (< 1ms) — fast pre-filter for obvious fraud patterns
2. **ML Ensemble** (< 10ms) — XGBoost + LightGBM + GNN probability blending

---

## 📊 Dataset

**PaySim** is a synthetic financial transaction simulator based on real mobile money data.

| Attribute        | Value                      |
|-----------------|----------------------------|
| Transactions     | 6.3 million                |
| Features         | 11 raw → 26 engineered     |
| Fraud Rate       | 0.13% (highly imbalanced)  |
| Time Period      | 744 steps (~1 month)       |
| Transaction Types| CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |

**Fraud only occurs in CASH_OUT and TRANSFER transactions.**

Download from Kaggle:
```bash
kaggle datasets download -d ntnu-eksamen/paysim1 -p data/ --unzip
```

---

## 🛠️ Installation

```bash
# 1. Clone / create project directory
mkdir fraud_detection && cd fraud_detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install PyTorch Geometric (CPU version)
pip install torch-scatter torch-sparse torch-cluster torch-geometric \
  -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# 5. Configure Kaggle API
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## 🚀 Usage

### Full Training Pipeline

```bash
# Full pipeline (download → build graph → train GNN + ensemble)
python train.py

# Skip Kaggle download (use existing CSV)
python train.py --skip-download

# Tabular ensemble only (faster, no GNN)
python train.py --skip-gnn

# Use a specific CSV file
python train.py --csv /path/to/transactions.csv
```

### Evaluate Saved Models

```bash
python evaluate.py
# Generates PR/ROC curves, confusion matrix, feature importance plots
# Saved to reports/
```

### Real-Time Stream Simulation

```bash
python inference.py
# Simulates a live transaction stream, prints alerts in real time
```

### REST API

```bash
python api.py
# API available at http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

### Streamlit Dashboard

```bash
streamlit run dashboard.py
# Dashboard at http://localhost:8501
```

---

## 🤖 Model Details

### Feature Engineering (26 features)

| Category       | Features |
|---------------|----------|
| Amount        | amount, log(amount), z-score, is_round |
| Balance       | diff_orig, diff_dest, ratio_orig, ratio_dest |
| Error Flags   | has_orig_error, has_dest_error, zero_before/after |
| Temporal      | hour_of_day, day_of_week, step_mod_24 |
| Account Agg.  | tx_count, total_sent, unique_destinations |
| Graph         | degree, pagerank, betweenness, clustering, local_fraud_rate |

### GraphSAGE Architecture

```
Input (in_channels=26+11) 
  → SAGEConv(→128) + BatchNorm + ReLU + Dropout
  → SAGEConv(→128) + BatchNorm + ReLU + Dropout  
  → SAGEConv(→64)  + BatchNorm + ReLU + Dropout
  → MLP(64→32→1) + Sigmoid
```

- **Loss:** Focal Loss (α=0.25, γ=2.0) handles extreme class imbalance
- **Optimizer:** AdamW + Cosine Annealing LR scheduler
- **Training:** Mini-batch NeighborLoader (15, 10, 5 neighbors per layer)
- **Early Stopping:** Patience=15 on Validation AUPRC

### Ensemble Weights

| Model     | Weight | Role |
|-----------|--------|------|
| XGBoost   | 35%    | Strong tabular learner |
| LightGBM  | 35%    | Gradient boosting |
| GraphSAGE | 30%    | Structural + neighborhood fraud patterns |

---

## 📈 Expected Results

| Metric    | Tabular Only | + GNN  |
|-----------|-------------|--------|
| AUPRC     | ~0.88       | ~0.93  |
| AUC-ROC   | ~0.97       | ~0.98  |
| F1 Score  | ~0.85       | ~0.89  |
| Precision | ~0.92       | ~0.94  |
| Recall    | ~0.79       | ~0.84  |

*Results depend on hardware, sampling, and hyperparameters.*

---

## 🌐 API Reference

### POST /score
Score a single transaction.

```json
{
  "step": 1,
  "type": "TRANSFER",
  "amount": 181374.00,
  "nameOrig": "C1231006815",
  "oldbalanceOrg": 181374.00,
  "newbalanceOrig": 0.00,
  "nameDest": "C553264065",
  "oldbalanceDest": 0.00,
  "newbalanceDest": 0.00
}
```

Response:
```json
{
  "fraud_probability": 0.9421,
  "is_fraud": true,
  "risk_level": "CRITICAL",
  "alert_reason": "Origin account fully drained; High-risk tx type (TRANSFER)",
  "latency_ms": 3.2
}
```

### POST /score/batch
Score up to 1000 transactions at once.

### GET /health
System health and running statistics.

---

## 📁 Project Structure

```
fraud_detection/
├── config.py               ← Central configuration
├── data_loader.py          ← Download, preprocess, split
├── graph_builder.py        ← Transaction graph construction
├── train_gnn.py            ← GNN training loop
├── train.py                ← Master training script
├── inference.py            ← Real-time scoring engine
├── api.py                  ← FastAPI REST endpoints
├── dashboard.py            ← Streamlit dashboard
├── evaluate.py             ← Evaluation plots
├── requirements.txt
├── models/
│   ├── gnn_model.py        ← GraphSAGE / GAT / GCN
│   └── ensemble_model.py   ← XGBoost + LightGBM stacking
├── data/                   ← Datasets
├── models/saved/           ← Checkpoints
├── logs/                   ← Training logs
└── reports/                ← Plots and JSON reports
```

---

## 🔑 Key Design Decisions

| Challenge | Solution |
|-----------|----------|
| Extreme class imbalance (0.13%) | SMOTE + Focal Loss + scale_pos_weight |
| Large graph (6M nodes) | Mini-batch NeighborLoader + graph sampling |
| Data leakage | Temporal train/val/test split (no shuffling) |
| Real-time latency | Two-stage: rules (< 1ms) + ML (< 10ms) |
| Overfitting | Dropout + BatchNorm + early stopping + L2 |
| False positives | Threshold tuning on validation F1 |

---

## 📜 License
MIT License — free for academic and commercial use.
