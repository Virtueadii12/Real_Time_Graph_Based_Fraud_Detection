# train_gnn.py
# ─────────────────────────────────────────────────────────────────────────────
# Full GNN training loop using mini-batch NeighborLoader.
# Handles class imbalance via Focal Loss + positive class weighting.
# ─────────────────────────────────────────────────────────────────────────────

import os
import logging
import time
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from pathlib import Path

from config import GNN_CONFIG, MODEL_DIR, SEED
from models.gnn_model import build_gnn_model, FocalLoss, compute_pos_weight

log = logging.getLogger(__name__)

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Mask-based train / val split for graph data
# ─────────────────────────────────────────────────────────────────────────────

def create_train_val_masks(data, val_ratio: float = 0.15):
    """
    Splits graph nodes into train / val masks.
    Stratified to keep fraud ratio stable.
    """
    from sklearn.model_selection import train_test_split

    n = data.num_nodes
    y = data.y.numpy()

    idx = np.arange(n)
    train_idx, val_idx = train_test_split(
        idx, test_size=val_ratio, stratify=y, random_state=SEED
    )

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True

    data.train_mask = train_mask
    data.val_mask   = val_mask
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Training Step
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)

        # Only compute loss on the seed nodes (first batch_size nodes)
        mask   = batch.train_mask[:batch.batch_size]
        logits = logits[:batch.batch_size][mask]
        y      = batch.y[:batch.batch_size][mask].float()

        if logits.numel() == 0:
            continue

        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Validation Step
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_gnn(model, data, mask, device):
    model.eval()
    data = data.to(device)
    logits = model(data.x, data.edge_index)
    proba  = torch.sigmoid(logits[mask]).cpu().numpy()
    y_true = data.y[mask].cpu().numpy()

    if y_true.sum() == 0:
        return 0.0, 0.0

    ap  = average_precision_score(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    return ap, auc


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Full Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_gnn(data, cfg: dict = None) -> torch.nn.Module:
    """
    Trains the GNN model on the PyG Data object.

    Parameters
    ----------
    data : torch_geometric.data.Data   (must have train_mask, val_mask, y, x)
    cfg  : GNN_CONFIG override dict

    Returns
    -------
    Trained model (moved to CPU, in eval mode)
    """
    cfg = cfg or GNN_CONFIG

    # Masks
    data = create_train_val_masks(data, val_ratio=0.15)

    # NeighborLoader for mini-batch training
    train_loader = NeighborLoader(
        data,
        num_neighbors  = [15, 10, 5],       # neighbors per layer
        batch_size     = cfg["batch_size"],
        input_nodes    = data.train_mask,
        shuffle        = True,
        num_workers    = 0,
    )

    # Model
    in_channels = data.x.shape[1]
    model = build_gnn_model(cfg["model_type"], in_channels, cfg).to(DEVICE)
    log.info(f"GNN model: {cfg['model_type']}  params={sum(p.numel() for p in model.parameters()):,}")

    # Loss: Focal Loss with class weight
    pos_weight = compute_pos_weight(data.y[data.train_mask]).to(DEVICE)
    criterion  = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)

    # Optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-5
    )

    # Training loop
    best_val_ap   = -1
    patience_ctr  = 0
    best_state    = None
    history       = {"train_loss": [], "val_ap": [], "val_auc": []}

    log.info("Starting GNN training …")
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_ap, val_auc = evaluate_gnn(model, data, data.val_mask, DEVICE)
        scheduler.step()

        history["train_loss"].append(loss)
        history["val_ap"].append(val_ap)
        history["val_auc"].append(val_auc)

        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch == 1:
            log.info(
                f"  Epoch {epoch:03d}/{cfg['epochs']}  "
                f"loss={loss:.4f}  val_AUPRC={val_ap:.4f}  "
                f"val_AUC={val_auc:.4f}  ({elapsed:.1f}s)"
            )

        # Early stopping on Val AUPRC
        if val_ap > best_val_ap:
            best_val_ap  = val_ap
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                log.info(f"  Early stopping at epoch {epoch}. Best Val AUPRC={best_val_ap:.4f}")
                break

    # Restore best weights
    model.load_state_dict(best_state)
    model = model.cpu().eval()

    # Save model
    save_path = MODEL_DIR / "gnn_model.pt"
    torch.save({
        "model_state"  : model.state_dict(),
        "in_channels"  : in_channels,
        "cfg"          : cfg,
        "best_val_ap"  : best_val_ap,
        "history"      : history,
    }, save_path)
    log.info(f"GNN saved → {save_path}")

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Inference: get node-level probabilities
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def gnn_predict_all(model, data) -> np.ndarray:
    """
    Returns fraud probabilities for ALL nodes in the graph.
    Uses full-graph inference (no mini-batching).
    """
    model.eval()
    logits = model(data.x, data.edge_index)
    return torch.sigmoid(logits).numpy()


def load_gnn_model(path: Path = None) -> torch.nn.Module:
    """Loads a saved GNN model."""
    path = path or (MODEL_DIR / "gnn_model.pt")
    ckpt = torch.load(path, map_location="cpu")
    model = build_gnn_model(ckpt["cfg"]["model_type"],
                             ckpt["in_channels"],
                             ckpt["cfg"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info(f"GNN loaded from {path}  (best val AUPRC={ckpt['best_val_ap']:.4f})")
    return model
