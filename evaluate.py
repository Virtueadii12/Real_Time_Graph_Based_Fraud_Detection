# evaluate.py
# ─────────────────────────────────────────────────────────────────────────────
# Standalone evaluation script — loads saved models and produces
# comprehensive reports with plots.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    average_precision_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from sklearn.calibration import calibration_curve
import joblib
from pathlib import Path

from config import MODEL_DIR, REPORT_DIR, SPLITS_PKL

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

plt.style.use("dark_background")
COLORS = {"fraud": "#ff4757", "legit": "#2ed573", "model": "#00d4ff"}


# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_roc_curves(y_true, y_proba, model_name="Ensemble",
                       save_path: Path = None):
    """Precision-Recall and ROC curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             facecolor="#0a0a1a")
    fig.suptitle(f"Model Evaluation — {model_name}", fontsize=16, color="white", y=1.02)

    # PR Curve
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    axes[0].plot(rec, prec, color=COLORS["model"], lw=2,
                 label=f"AUPRC = {ap:.4f}")
    baseline = y_true.mean()
    axes[0].axhline(baseline, color="gray", ls="--", lw=1,
                    label=f"Baseline (random) = {baseline:.4f}")
    axes[0].set_xlabel("Recall",    color="white")
    axes[0].set_ylabel("Precision", color="white")
    axes[0].set_title("Precision–Recall Curve", color="white")
    axes[0].legend()
    axes[0].set_facecolor("#111122")
    axes[0].tick_params(colors="white")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_roc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color=COLORS["model"], lw=2,
                 label=f"AUC-ROC = {auc_roc:.4f}")
    axes[1].plot([0,1],[0,1], color="gray", ls="--", lw=1, label="Random")
    axes[1].set_xlabel("False Positive Rate", color="white")
    axes[1].set_ylabel("True Positive Rate",  color="white")
    axes[1].set_title("ROC Curve", color="white")
    axes[1].legend()
    axes[1].set_facecolor("#111122")
    axes[1].tick_params(colors="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#0a0a1a")
        log.info(f"PR/ROC plot saved → {save_path}")
    plt.show()
    return fig


def plot_confusion_matrix(y_true, y_pred, save_path: Path = None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0a0a1a")
    sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                cmap="Blues",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"])
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual",    color="white")
    ax.set_title("Confusion Matrix", color="white")
    ax.set_facecolor("#111122")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0a1a")
    plt.show()
    return fig


def plot_score_distribution(y_true, y_proba, threshold: float = 0.5,
                             save_path: Path = None):
    """Fraud probability histogram split by true label."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0a0a1a")
    ax.set_facecolor("#111122")

    bins = np.linspace(0, 1, 50)
    ax.hist(y_proba[y_true==0], bins=bins, alpha=0.6,
            color=COLORS["legit"], label="Legitimate", density=True)
    ax.hist(y_proba[y_true==1], bins=bins, alpha=0.6,
            color=COLORS["fraud"], label="Fraud", density=True)
    ax.axvline(threshold, color="yellow", ls="--", lw=2,
               label=f"Threshold = {threshold:.2f}")
    ax.set_xlabel("Predicted Fraud Probability", color="white")
    ax.set_ylabel("Density", color="white")
    ax.set_title("Score Distribution by True Label", color="white")
    ax.legend()
    ax.tick_params(colors="white")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0a1a")
    plt.show()
    return fig


def plot_threshold_analysis(y_true, y_proba, save_path: Path = None):
    """F1, Precision, Recall vs threshold."""
    thresholds = np.linspace(0.1, 0.95, 80)
    metrics = {"precision": [], "recall": [], "f1": []}

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred==1) & (y_true==1)).sum()
        fp = ((y_pred==1) & (y_true==0)).sum()
        fn = ((y_pred==0) & (y_true==1)).sum()
        p  = tp / (tp + fp + 1e-9)
        r  = tp / (tp + fn + 1e-9)
        f1 = 2*p*r/(p+r+1e-9)
        metrics["precision"].append(p)
        metrics["recall"].append(r)
        metrics["f1"].append(f1)

    best_t = thresholds[np.argmax(metrics["f1"])]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0a0a1a")
    ax.set_facecolor("#111122")
    ax.plot(thresholds, metrics["precision"], color="#ffd32a", lw=2, label="Precision")
    ax.plot(thresholds, metrics["recall"],    color="#2ed573", lw=2, label="Recall")
    ax.plot(thresholds, metrics["f1"],        color="#00d4ff", lw=2.5, label="F1")
    ax.axvline(best_t, color="white", ls="--", lw=1.5,
               label=f"Best threshold = {best_t:.2f}")
    ax.set_xlabel("Classification Threshold", color="white")
    ax.set_ylabel("Score", color="white")
    ax.set_title("Precision / Recall / F1 vs Threshold", color="white")
    ax.legend()
    ax.tick_params(colors="white")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0a1a")
    plt.show()
    return fig


def plot_feature_importance(ensemble, feature_cols: list, save_path: Path = None):
    """Side-by-side XGBoost and LightGBM feature importance."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor="#0a0a1a")

    for ax, (model, name) in zip(axes, [
        (ensemble.xgb_model, "XGBoost"),
        (ensemble.lgb_model, "LightGBM"),
    ]):
        fi = model.feature_importances_
        df = pd.DataFrame({"Feature": feature_cols, "Importance": fi})
        df = df.sort_values("Importance", ascending=True).tail(15)

        ax.barh(df["Feature"], df["Importance"],
                color=COLORS["model"], alpha=0.85)
        ax.set_title(f"{name} Feature Importance", color="white")
        ax.set_xlabel("Importance", color="white")
        ax.set_facecolor("#111122")
        ax.tick_params(colors="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0a1a")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────

def run_full_evaluation():
    """Load saved models and produce all evaluation plots."""
    log.info("Loading data and models for evaluation …")

    splits   = joblib.load(SPLITS_PKL)
    ensemble = joblib.load(MODEL_DIR / "ensemble.pkl")

    X_test  = splits["X_test"]
    y_test  = splits["y_test"]

    y_proba = ensemble.predict_proba(X_test)
    y_pred  = ensemble.predict(X_test)

    # Print classification report
    print(classification_report(y_test, y_pred, target_names=["Legit","Fraud"]))

    # Plots
    plot_pr_roc_curves(y_test, y_proba,
                       save_path=REPORT_DIR / "pr_roc_curves.png")
    plot_confusion_matrix(y_test, y_pred,
                          save_path=REPORT_DIR / "confusion_matrix.png")
    plot_score_distribution(y_test, y_proba,
                             threshold=ensemble.threshold,
                             save_path=REPORT_DIR / "score_distribution.png")
    plot_threshold_analysis(y_test, y_proba,
                            save_path=REPORT_DIR / "threshold_analysis.png")
    plot_feature_importance(ensemble, splits["feature_cols"],
                             save_path=REPORT_DIR / "feature_importance.png")

    log.info(f"All plots saved to {REPORT_DIR}")


if __name__ == "__main__":
    run_full_evaluation()
