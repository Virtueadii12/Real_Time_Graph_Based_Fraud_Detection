# models/gnn_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Graph Neural Network models for fraud detection.
# Implements GraphSAGE, GAT, and GCN with an MLP classifier head.
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    SAGEConv, GATConv, GCNConv,
    BatchNorm, global_mean_pool,
)
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import numpy as np
import logging
from typing import Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GraphSAGE Fraud Detector  (Primary model)
# ─────────────────────────────────────────────────────────────────────────────

class GraphSAGEFraudDetector(nn.Module):
    """
    Multi-layer GraphSAGE encoder + MLP classifier.

    Architecture
    ------------
    Input  →  [GraphSAGE × num_layers]  →  Dropout  →  MLP head  →  sigmoid
    
    Each SAGE layer aggregates neighbor information using mean/max/LSTM pooling.
    BatchNorm after each layer stabilises training on the highly skewed
    fraud distribution.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 aggregation: str = "mean"):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregation))
        self.bns.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregation))
            self.bns.append(BatchNorm(hidden_channels))

        # Last GNN layer
        self.convs.append(SAGEConv(hidden_channels, hidden_channels // 2, aggr=aggregation))
        self.bns.append(BatchNorm(hidden_channels // 2))

        self.dropout = nn.Dropout(dropout)

        # MLP Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns node embeddings before the classifier."""
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Returns raw logits (apply sigmoid externally or use BCEWithLogitsLoss).
        """
        h = self.encode(x, edge_index)
        out = self.classifier(h).squeeze(-1)
        return out

    def predict_proba(self, x: torch.Tensor,
                      edge_index: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            probs  = torch.sigmoid(logits).cpu().numpy()
        return probs


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Graph Attention Network (GAT)
# ─────────────────────────────────────────────────────────────────────────────

class GATFraudDetector(nn.Module):
    """
    Multi-head GAT.  Attention weights can be inspected for explainability.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 num_layers: int = 3,
                 heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads,
                                   dropout=dropout, concat=True))
        self.bns.append(BatchNorm(hidden_channels * heads))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                       heads=heads, dropout=dropout, concat=True))
            self.bns.append(BatchNorm(hidden_channels * heads))

        self.convs.append(GATConv(hidden_channels * heads, hidden_channels // 2,
                                   heads=1, dropout=dropout, concat=False))
        self.bns.append(BatchNorm(hidden_channels // 2))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = self.dropout(x)
        return self.classifier(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GCN Baseline
# ─────────────────────────────────────────────────────────────────────────────

class GCNFraudDetector(nn.Module):
    """Vanilla GCN baseline."""

    def __init__(self, in_channels, hidden_channels=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, hidden_channels // 2))
        self.bns.append(BatchNorm(hidden_channels // 2))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        return self.classifier(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Model Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_gnn_model(model_type: str, in_channels: int, cfg: dict) -> nn.Module:
    """
    Factory function — returns the requested GNN model.

    Parameters
    ----------
    model_type   : "GraphSAGE" | "GAT" | "GCN"
    in_channels  : number of input node features
    cfg          : GNN_CONFIG dict from config.py
    """
    if model_type == "GraphSAGE":
        return GraphSAGEFraudDetector(
            in_channels     = in_channels,
            hidden_channels = cfg["hidden_channels"],
            num_layers      = cfg["num_layers"],
            dropout         = cfg["dropout"],
            aggregation     = cfg["aggregation"],
        )
    elif model_type == "GAT":
        return GATFraudDetector(
            in_channels     = in_channels,
            hidden_channels = cfg["hidden_channels"],
            num_layers      = cfg["num_layers"],
            heads           = cfg["heads"],
            dropout         = cfg["dropout"],
        )
    elif model_type == "GCN":
        return GCNFraudDetector(
            in_channels     = in_channels,
            hidden_channels = cfg["hidden_channels"],
            num_layers      = cfg["num_layers"],
            dropout         = cfg["dropout"],
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Focal Loss (handles extreme class imbalance better than BCE)
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Reduces the relative loss for well-classified examples (easy negatives),
    focusing training on hard misclassified examples (rare frauds).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha      = alpha
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=self.pos_weight, reduction="none"
        )
        p_t  = torch.exp(-bce)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Training Utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_fraud_ratio(y: torch.Tensor) -> float:
    return y.float().mean().item()


def compute_pos_weight(y: torch.Tensor) -> torch.Tensor:
    n_pos = y.sum().float()
    n_neg = (y == 0).sum().float()
    return (n_neg / (n_pos + 1e-9)).unsqueeze(0)
