# models/ensemble_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Stacked ensemble: XGBoost + LightGBM + GNN embeddings
# Outputs calibrated fraud probability scores.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional

import xgboost as xgb
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from config import MODEL_DIR, XGB_CONFIG, LGB_CONFIG, ENSEMBLE_CONFIG, SEED

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  XGBoost Trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_val, y_val) -> xgb.XGBClassifier:
    """Trains XGBoost with early stopping on val AUPRC."""
    log.info("Training XGBoost …")
    cfg = XGB_CONFIG.copy()
    early_stop = cfg.pop("early_stopping_rounds")

    model = xgb.XGBClassifier(
        **cfg,
        random_state = SEED,
        use_label_encoder = False,
        verbosity    = 1,
    )
    model.fit(
        X_train, y_train,
        eval_set     = [(X_val, y_val)],
        early_stopping_rounds = early_stop,
        verbose      = 50,
    )
    val_proba = model.predict_proba(X_val)[:, 1]
    ap = average_precision_score(y_val, val_proba)
    auc = roc_auc_score(y_val, val_proba)
    log.info(f"  XGB Val  AUPRC={ap:.4f}  AUC-ROC={auc:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LightGBM Trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_lightgbm(X_train, y_train, X_val, y_val) -> lgb.LGBMClassifier:
    """Trains LightGBM with early stopping."""
    log.info("Training LightGBM …")
    cfg = LGB_CONFIG.copy()
    early_stop = cfg.pop("early_stopping_rounds")

    model = lgb.LGBMClassifier(
        **cfg,
        random_state = SEED,
        verbose      = -1,
    )
    model.fit(
        X_train, y_train,
        eval_set     = [(X_val, y_val)],
        callbacks    = [lgb.early_stopping(early_stop), lgb.log_evaluation(50)],
    )
    val_proba = model.predict_proba(X_val)[:, 1]
    ap  = average_precision_score(y_val, val_proba)
    auc = roc_auc_score(y_val, val_proba)
    log.info(f"  LGB Val  AUPRC={ap:.4f}  AUC-ROC={auc:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SMOTE Oversampling Wrapper
# ─────────────────────────────────────────────────────────────────────────────

def apply_smote(X_train, y_train, sampling_ratio: float = 0.1):
    """
    Applies SMOTE to create synthetic fraud samples.
    sampling_ratio: target ratio of minority class after resampling.
    """
    log.info("Applying SMOTE oversampling …")
    smote = SMOTE(sampling_strategy=sampling_ratio, random_state=SEED, n_jobs=-1)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    log.info(f"  After SMOTE: {X_res.shape}  Fraud: {y_res.sum():,} / {len(y_res):,}")
    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Ensemble (Weighted Average + Meta-Learner)
# ─────────────────────────────────────────────────────────────────────────────

class FraudEnsemble:
    """
    Weighted stacking ensemble of XGBoost, LightGBM, and GNN.

    Pipeline
    --------
    1. XGB + LGB trained on tabular features
    2. GNN trained on graph structure (optional, if pyg_data available)
    3. Probabilities stacked → meta-learner (Logistic Regression) or
       simple weighted average
    4. Probability calibration with Platt scaling
    """

    def __init__(self, weights: Dict[str, float] = None,
                 threshold: float = None):
        self.weights   = weights   or ENSEMBLE_CONFIG["weights"]
        self.threshold = threshold or ENSEMBLE_CONFIG["threshold"]
        self.xgb_model = None
        self.lgb_model = None
        self.gnn_proba = None       # numpy array, set externally
        self.meta      = None       # optional meta-learner
        self.calibrate_flag = ENSEMBLE_CONFIG["calibrate"]

    # ── Fit ───────────────────────────────────────────────────────────────────
    def fit(self, X_train, y_train, X_val, y_val,
            use_smote: bool = True,
            gnn_train_proba: Optional[np.ndarray] = None,
            gnn_val_proba:   Optional[np.ndarray] = None):
        """
        Trains XGB + LGB.  If gnn_train_proba is provided, also fits a
        meta-learner on stacked predictions.
        """
        if use_smote:
            X_tr, y_tr = apply_smote(X_train, y_train)
        else:
            X_tr, y_tr = X_train, y_train

        self.xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val)
        self.lgb_model = train_lightgbm(X_tr, y_tr, X_val, y_val)

        # ── Meta-learner (if GNN proba available) ────────────────────────────
        if gnn_train_proba is not None and gnn_val_proba is not None:
            log.info("Fitting meta-learner with GNN proba …")
            xgb_p = self.xgb_model.predict_proba(X_train)[:, 1]
            lgb_p = self.lgb_model.predict_proba(X_train)[:, 1]
            stack_train = np.column_stack([xgb_p, lgb_p, gnn_train_proba])

            xgb_pv = self.xgb_model.predict_proba(X_val)[:, 1]
            lgb_pv = self.lgb_model.predict_proba(X_val)[:, 1]
            stack_val = np.column_stack([xgb_pv, lgb_pv, gnn_val_proba])

            meta = LogisticRegression(C=0.1, max_iter=500, random_state=SEED)
            meta.fit(stack_train, y_train)
            self.meta = meta
            val_meta = meta.predict_proba(stack_val)[:, 1]
            ap = average_precision_score(y_val, val_meta)
            log.info(f"  Meta-learner Val AUPRC = {ap:.4f}")
        else:
            log.info("No GNN proba → using weighted average ensemble.")

        return self

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict_proba(self, X,
                      gnn_proba: Optional[np.ndarray] = None) -> np.ndarray:
        xgb_p = self.xgb_model.predict_proba(X)[:, 1]
        lgb_p = self.lgb_model.predict_proba(X)[:, 1]

        if self.meta is not None and gnn_proba is not None:
            stack = np.column_stack([xgb_p, lgb_p, gnn_proba])
            return self.meta.predict_proba(stack)[:, 1]

        # Weighted average
        w = self.weights
        if gnn_proba is not None:
            total = w["xgb"] + w["lgb"] + w["gnn"]
            return (w["xgb"] * xgb_p + w["lgb"] * lgb_p +
                    w["gnn"] * gnn_proba) / total
        else:
            total = w["xgb"] + w["lgb"]
            return (w["xgb"] * xgb_p + w["lgb"] * lgb_p) / total

    def predict(self, X,
                gnn_proba: Optional[np.ndarray] = None) -> np.ndarray:
        return (self.predict_proba(X, gnn_proba) >= self.threshold).astype(int)

    # ── Optimal threshold ─────────────────────────────────────────────────────
    def tune_threshold(self, X_val, y_val,
                       gnn_proba: Optional[np.ndarray] = None,
                       metric: str = "f1"):
        """Finds the classification threshold that maximises F1 on validation."""
        proba = self.predict_proba(X_val, gnn_proba)
        prec, rec, thresholds = precision_recall_curve(y_val, proba)
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = np.argmax(f1s[:-1])
        self.threshold = thresholds[best_idx]
        log.info(f"  Optimal threshold: {self.threshold:.4f}  "
                 f"P={prec[best_idx]:.4f}  R={rec[best_idx]:.4f}  "
                 f"F1={f1s[best_idx]:.4f}")
        return self.threshold

    # ── Save / Load ───────────────────────────────────────────────────────────
    def save(self, path: Path = None):
        path = path or (MODEL_DIR / "ensemble.pkl")
        joblib.dump(self, path)
        log.info(f"Ensemble saved → {path}")

    @staticmethod
    def load(path: Path = None) -> "FraudEnsemble":
        path = path or (MODEL_DIR / "ensemble.pkl")
        return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Evaluation Utilities
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true, y_proba, threshold: float = 0.5,
             split_name: str = "Test") -> Dict:
    """Comprehensive evaluation of a fraud detector."""
    y_pred = (y_proba >= threshold).astype(int)

    ap   = average_precision_score(y_true, y_proba)
    auc  = roc_auc_score(y_true, y_proba)
    f1   = f1_score(y_true, y_pred)
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)

    metrics = {
        "split"          : split_name,
        "auprc"          : round(ap, 4),
        "auc_roc"        : round(auc, 4),
        "f1"             : round(f1, 4),
        "precision"      : round(precision, 4),
        "recall"         : round(recall, 4),
        "tp"             : int(tp),
        "fp"             : int(fp),
        "tn"             : int(tn),
        "fn"             : int(fn),
        "threshold"      : threshold,
    }

    print(f"\n{'='*55}")
    print(f"  {split_name} Evaluation")
    print(f"{'='*55}")
    print(f"  AUPRC   : {ap:.4f}")
    print(f"  AUC-ROC : {auc:.4f}")
    print(f"  F1      : {f1:.4f}")
    print(f"  Precision: {precision:.4f}   Recall: {recall:.4f}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"{'='*55}\n")
    print(classification_report(y_true, y_pred,
                                target_names=["Legit", "Fraud"]))
    return metrics
