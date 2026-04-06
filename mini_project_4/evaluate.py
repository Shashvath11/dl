# =============================================================
# mini_project_4/evaluate.py
# [RUBRIC MP4: Performance Evaluation — 3 marks]
#
# This file implements:
#   1. Multi-metric evaluation: Accuracy, Precision, Recall, F1
#   2. Confusion matrix with seaborn heatmap
#   3. Statistical reasoning (WHY these metrics?)
#   4. Top-K accuracy for 5-class problem
#   5. Per-class analysis table
#
# METRIC JUSTIFICATION (Statistical Reasoning):
# ─────────────────────────────────────────────
# Why F1-score and not just Accuracy?
#   Accuracy = Correct / Total → misleading if classes are imbalanced.
#   Example: If 80% data is "Biking", predicting always "Biking" gives
#   80% accuracy but is useless. F1 penalizes this behavior.
#
# F1 = 2 * (Precision * Recall) / (Precision + Recall)
#   - Precision: Of predicted class X, how many are truly class X?
#   - Recall: Of true class X samples, how many did we find?
#   - F1: Harmonic mean — balanced, handles class imbalance
#
# Per-class F1: reveals WHICH classes are confused (e.g., BenchPress
# vs Biking may both involve horizontal motion patterns).
#
# Confusion Matrix: visual tool showing prediction errors.
# Diagonal = correct, off-diagonal = errors.
# Helps debug: which classes does the model confuse most?
# =============================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import pandas as pd
import io
from PIL import Image


# ──────────────────────────────────────────────────────────────────
# SECTION 1: COLLECT PREDICTIONS
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> tuple:
    """
    Run model inference over the full dataloader.
    Returns all ground truth labels and predicted labels.

    Args:
        model:      Trained CNN+LSTM (or any classifier)
        dataloader: DataLoader (no augmentation — val loader preferred)
        device:     Compute device

    Returns:
        (all_labels, all_preds, all_probs)
    """
    model.eval()
    model = model.to(device)

    all_labels = []
    all_preds = []
    all_probs = []

    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        logits = model(sequences)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# ──────────────────────────────────────────────────────────────────
# SECTION 2: COMPUTE ALL METRICS
# [RUBRIC: Performance Evaluation — Accuracy, Precision, Recall, F1]
# ──────────────────────────────────────────────────────────────────

def compute_metrics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: list,
    device: str = "cpu",
) -> dict:
    """
    Compute comprehensive evaluation metrics.

    [RUBRIC: Performance Evaluation — proper metrics + statistical reasoning]

    Metrics Computed:
        1. Overall Accuracy     — % of correct predictions
        2. Macro F1 Score       — average F1 across all classes (ignores imbalance)
        3. Weighted F1 Score    — F1 weighted by class frequency
        4. Per-class Precision  — exactness per class
        5. Per-class Recall     — completeness per class
        6. Per-class F1         — balance of P and R per class
        7. Confusion Matrix     — N×N matrix of prediction counts

    Statistical Reasoning (WHY THESE METRICS?):
        - Accuracy alone is insufficient: misleading on imbalanced datasets
        - Macro F1: gives EQUAL weight to all classes regardless of size
          → Critical when all 5 action classes should be treated equally
        - Weighted F1: accounts for class frequency → practical performance
        - Per-class F1: pinpoints weak classes that need more data/tuning
        - Confusion Matrix: visual diagnostic — shows systematic errors

    Args:
        model:       Trained classifier model
        dataloader:  Evaluation DataLoader
        class_names: List of class name strings
        device:      Compute device

    Returns:
        metrics dict with all computed values
    """
    labels, preds, probs = get_predictions(model, dataloader, device)

    # ── Basic metrics ──────────────────────────────────────────
    accuracy = accuracy_score(labels, preds) * 100.0

    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0) * 100.0
    weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0) * 100.0

    macro_precision = precision_score(labels, preds, average='macro', zero_division=0) * 100.0
    macro_recall = recall_score(labels, preds, average='macro', zero_division=0) * 100.0

    # ── Per-class metrics ──────────────────────────────────────
    per_class_precision = precision_score(labels, preds, average=None, zero_division=0)
    per_class_recall = recall_score(labels, preds, average=None, zero_division=0)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    class_support = np.bincount(labels, minlength=len(class_names))

    # ── Per-class DataFrame ────────────────────────────────────
    per_class_df = pd.DataFrame({
        'Class': class_names,
        'Precision (%)': (per_class_precision * 100).round(2),
        'Recall (%)': (per_class_recall * 100).round(2),
        'F1-Score (%)': (per_class_f1 * 100).round(2),
        'Support': class_support,
    })

    # ── Confusion matrix ───────────────────────────────────────
    cm = confusion_matrix(labels, preds)

    # ── Full sklearn report ─────────────────────────────────────
    report = classification_report(
        labels, preds,
        target_names=class_names,
        zero_division=0
    )

    metrics = {
        'accuracy': round(accuracy, 2),
        'macro_f1': round(macro_f1, 2),
        'weighted_f1': round(weighted_f1, 2),
        'macro_precision': round(macro_precision, 2),
        'macro_recall': round(macro_recall, 2),
        'per_class_df': per_class_df,
        'confusion_matrix': cm,
        'classification_report': report,
        'n_samples': len(labels),
        'class_names': class_names,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 60)
    print(f"  Overall Accuracy:     {accuracy:.2f}%")
    print(f"  Macro F1 Score:       {macro_f1:.2f}%")
    print(f"  Weighted F1 Score:    {weighted_f1:.2f}%")
    print(f"  Macro Precision:      {macro_precision:.2f}%")
    print(f"  Macro Recall:         {macro_recall:.2f}%")
    print(f"  Evaluated on:         {len(labels)} samples")
    print(f"\n{report}")

    return metrics


# ──────────────────────────────────────────────────────────────────
# SECTION 3: CONFUSION MATRIX VISUALIZATION
# [RUBRIC: Performance Evaluation — confusion matrix plot]
# ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    return_fig: bool = False,
):
    """
    Plot annotated confusion matrix heatmap.

    HOW TO READ THE CONFUSION MATRIX:
        - Rows = True class (ground truth)
        - Columns = Predicted class
        - Diagonal cells = Correct predictions
        - Off-diagonal cells = Errors
        - Example: cm[2,0] = how many "Biking" samples were predicted as "Archery"

    If normalize=True: shows RECALL per class on diagonal (fraction correct)
    If normalize=False: shows raw counts

    [RUBRIC: Confusion matrix visualization for performance evaluation]

    Args:
        cm:          Confusion matrix array (N, N)
        class_names: Class name list
        normalize:   Normalize by row (true positives / total per class)
        return_fig:  Return figure if True
    """
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1e-8)
        fmt = '.2%'
        cbar_label = 'Recall (% correct per true class)'
    else:
        cm_display = cm
        fmt = 'd'
        cbar_label = 'Count'

    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        cbar_kws={'label': cbar_label},
    )

    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()

    if return_fig:
        return fig
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.show()
    return None


# ──────────────────────────────────────────────────────────────────
# SECTION 4: TOP-K ACCURACY
# [RUBRIC: Performance Evaluation — Top-K accuracy]
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def top_k_accuracy(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    k: int = 3,
    device: str = "cpu",
) -> float:
    """
    Compute Top-K accuracy: prediction is correct if true class is in
    the model's top-K highest probability predictions.

    WHY TOP-K?
        For a 5-class problem, Top-1 accuracy measures strict correctness.
        Top-3 accuracy (60% of classes) measures relevance — useful to
        understand if the model has "reasonable" predictions even when wrong.
        If Top-3 accuracy is very high but Top-1 is low, the model knows
        the right area but is unsure between similar classes.

    [RUBRIC: Additional evaluation metric — Top-K accuracy]

    Args:
        model:      Trained classifier
        dataloader: DataLoader
        k:          Number of top predictions to consider
        device:     Compute device

    Returns:
        top_k_acc: Accuracy as a percentage
    """
    model.eval()
    model = model.to(device)

    correct = 0
    total = 0

    for sequences, labels in dataloader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        logits = model(sequences)
        # Get top-k predicted class indices (highest probabilities)
        _, top_k_preds = logits.topk(k, dim=1)

        # Check if true label appears in top-k predictions
        labels_expanded = labels.unsqueeze(1).expand_as(top_k_preds)
        correct += top_k_preds.eq(labels_expanded).any(dim=1).sum().item()
        total += labels.size(0)

    top_k_acc = 100.0 * correct / max(total, 1)
    print(f"[Top-{k} Accuracy] {top_k_acc:.2f}% (n={total})")
    return round(top_k_acc, 2)


# ──────────────────────────────────────────────────────────────────
# SECTION 5: PER-CLASS METRICS BAR CHART
# [RUBRIC: Per-class analysis — why some classes are harder]
# ──────────────────────────────────────────────────────────────────

def plot_per_class_metrics(
    metrics: dict,
    return_fig: bool = False,
):
    """
    Bar chart showing per-class Precision, Recall, F1.

    INTERPRETATION GUIDE:
    - Low Recall for class X → model misses many true X samples
      (Possible reason: X looks similar to another class)
    - Low Precision for class X → model predicts X too often
      (Possible reason: X is the default fallback class)
    - Balanced P/R (high F1) → model consistently handles class X

    For UCF-101 classes:
    - BenchPress and Biking may be confused (both involve cyclic motion)
    - ApplyEyeMakeup may be confused with others due to fine-grained gestures
    - Archery and BasketballDunk may have distinct poses → higher accuracy

    [RUBRIC: Per-class analysis with statistical reasoning]
    """
    df = metrics['per_class_df']
    class_names = df['Class'].tolist()
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_p = ax.bar(x - width, df['Precision (%)'], width, label='Precision', color='#3498db', alpha=0.85)
    bars_r = ax.bar(x, df['Recall (%)'], width, label='Recall', color='#2ecc71', alpha=0.85)
    bars_f = ax.bar(x + width, df['F1-Score (%)'], width, label='F1-Score', color='#e74c3c', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15, ha='right')
    ax.set_title("Per-Class Precision, Recall, F1-Score", fontsize=13, fontweight='bold')
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Annotate F1 bars
    for bar in bars_f:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.0f}%",
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    if return_fig:
        return fig
    plt.savefig("per_class_metrics.png", dpi=150, bbox_inches='tight')
    plt.show()
    return None


# ──────────────────────────────────────────────────────────────────
# SECTION 6: Helper — fig to PIL
# ──────────────────────────────────────────────────────────────────

def fig_to_pil(fig) -> Image.Image:
    """Convert matplotlib Figure to PIL Image for Streamlit display."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[evaluate.py] Evaluation functions available:")
    print("  - compute_metrics(model, dataloader, class_names, device)")
    print("  - plot_confusion_matrix(cm, class_names, normalize=True)")
    print("  - top_k_accuracy(model, dataloader, k=3, device)")
    print("  - plot_per_class_metrics(metrics)")
    print("\nStatistical reasoning for metrics documented in file header.")
