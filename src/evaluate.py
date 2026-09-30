"""
Evaluation script for the Fruit Image Classifier.

Loads a trained model and computes:
  • Test accuracy, precision, recall, F1-score (per-class & macro)
  • Confusion matrix (saved as PNG)
  • Per-class accuracy bar chart
  • Training vs validation accuracy / loss plots (saved as PNG)

Usage:
    python src/evaluate.py [--data-dir data] [--model-dir models]
                           [--output-dir outputs]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ── Style ─────────────────────────────────────────────────────────────────
DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
ACCENT    = "#e94560"
BLUE      = "#0f3460"
TEXT      = "#eaeaea"
PALETTE   = ["#4cc9f0", "#f72585", "#7209b7", "#3a0ca3", "#4361ee", "#4cc9f0"]
FRUIT_COLORS = {
    "apple":  "#e63946",
    "banana": "#f4d03f",
    "grape":  "#8e44ad",
    "mango":  "#f39c12",
    "orange": "#e67e22",
}

FRUIT_EMOJI = {
    "apple":  "🍎",
    "banana": "🍌",
    "grape":  "🍇",
    "mango":  "🥭",
    "orange": "🍊",
}


def _setup_style():
    """Apply a dark, premium matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    "#2a2a4a",
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "axes.titlesize":    14,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "axes.grid":         True,
        "grid.color":        "#2a2a4a",
        "grid.linewidth":    0.6,
        "xtick.color":       TEXT,
        "ytick.color":       TEXT,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.facecolor":  PANEL_BG,
        "legend.edgecolor":  "#2a2a4a",
        "legend.labelcolor": TEXT,
        "legend.fontsize":   10,
        "text.color":        TEXT,
        "figure.dpi":        150,
    })


def load_model_and_classes(model_dir: str):
    """Load the trained Keras model and class-name list."""
    model_path = os.path.join(model_dir, "fruit_classifier.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "best_model.keras")
    model = keras.models.load_model(model_path)

    with open(os.path.join(model_dir, "class_names.json")) as f:
        class_names = json.load(f)
    return model, class_names


def get_test_dataset(data_dir: str):
    """Load the test split as a tf.data.Dataset."""
    return keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )


def collect_predictions(model, dataset):
    """Run inference on the full dataset, return (y_true, y_pred) as 1-D arrays."""
    y_true_all, y_pred_all = [], []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_pred_all.append(np.argmax(preds, axis=1))
        y_true_all.append(np.argmax(labels.numpy(), axis=1))
    return np.concatenate(y_true_all), np.concatenate(y_pred_all)


# ── Plots ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Generate and save a styled confusion-matrix heatmap."""
    _setup_style()
    cm = confusion_matrix(y_true, y_pred)

    labels = [f"{FRUIT_EMOJI.get(c, '')} {c.capitalize()}" for c in class_names]

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold", pad=15)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else TEXT
            weight = "bold" if i == j else "normal"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=13, fontweight=weight)

    # Highlight diagonal
    for i in range(len(class_names)):
        rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                              fill=False, edgecolor=ACCENT, linewidth=2)
        ax.add_patch(rect)

    plt.tight_layout(pad=1.5)
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved → {path}")


def plot_training_history(history: dict, output_dir: str):
    """Plot training vs validation accuracy and loss — dark premium style."""
    _setup_style()
    epochs_range = range(1, len(history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Training History", fontsize=18, fontweight="bold",
                 color=TEXT, y=1.02)

    # ── Accuracy ──
    ax = axes[0]
    ax.plot(epochs_range, history["accuracy"],
            color="#4cc9f0", linewidth=2.5, marker="o", markersize=5,
            label="Train Accuracy")
    ax.plot(epochs_range, history["val_accuracy"],
            color=ACCENT, linewidth=2.5, marker="s", markersize=5,
            linestyle="--", label="Val Accuracy")
    best_val_ep = int(np.argmax(history["val_accuracy"])) + 1
    best_val_acc = max(history["val_accuracy"])
    ax.axvline(best_val_ep, color=ACCENT, linestyle=":", alpha=0.6, linewidth=1.5)
    ax.annotate(f"Best val\n{best_val_acc:.2%}",
                xy=(best_val_ep, best_val_acc),
                xytext=(best_val_ep + 0.4, best_val_acc - 0.05),
                color=ACCENT, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))
    ax.set_title("Accuracy", pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="lower right")
    ax.set_xticks(list(epochs_range))

    # ── Loss ──
    ax = axes[1]
    ax.plot(epochs_range, history["loss"],
            color="#4cc9f0", linewidth=2.5, marker="o", markersize=5,
            label="Train Loss")
    ax.plot(epochs_range, history["val_loss"],
            color=ACCENT, linewidth=2.5, marker="s", markersize=5,
            linestyle="--", label="Val Loss")
    best_val_loss_ep = int(np.argmin(history["val_loss"])) + 1
    best_val_loss = min(history["val_loss"])
    ax.axvline(best_val_loss_ep, color=ACCENT, linestyle=":", alpha=0.6, linewidth=1.5)
    ax.annotate(f"Best val\n{best_val_loss:.4f}",
                xy=(best_val_loss_ep, best_val_loss),
                xytext=(best_val_loss_ep + 0.4, best_val_loss + 0.05),
                color=ACCENT, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2))
    ax.set_title("Loss", pad=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend(loc="upper right")
    ax.set_xticks(list(epochs_range))

    plt.tight_layout(pad=2)
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved → {path}")


def plot_per_class_metrics(y_true, y_pred, class_names, output_dir):
    """Bar chart of per-class precision, recall, F1."""
    _setup_style()
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names))
    )

    x = np.arange(len(class_names))
    width = 0.26
    labels = [f"{FRUIT_EMOJI.get(c, '')} {c.capitalize()}" for c in class_names]
    colors = [FRUIT_COLORS.get(c, "#4cc9f0") for c in class_names]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(DARK_BG)

    bars_p = ax.bar(x - width, prec, width, label="Precision",
                    color="#4cc9f0", alpha=0.85, edgecolor="none")
    bars_r = ax.bar(x,        rec,  width, label="Recall",
                    color="#f72585", alpha=0.85, edgecolor="none")
    bars_f = ax.bar(x + width, f1,  width, label="F1-Score",
                    color="#4ade80", alpha=0.85, edgecolor="none")

    # Value labels on top of each bar
    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=8.5, color=TEXT)

    # Colour-coded fruit name ticks
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    for tick_label, cls in zip(ax.get_xticklabels(), class_names):
        tick_label.set_color(FRUIT_COLORS.get(cls, TEXT))

    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Per-Class Metrics: Precision · Recall · F1", pad=12)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")

    # Accuracy summary annotation
    acc = accuracy_score(y_true, y_pred)
    ax.text(0.99, 0.97, f"Overall Accuracy: {acc:.2%}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color=ACCENT, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BLUE, alpha=0.7))

    plt.tight_layout(pad=2)
    path = os.path.join(output_dir, "per_class_metrics.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Per-class metrics chart saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def evaluate(
    data_dir: str = "data",
    model_dir: str = "models",
    output_dir: str = "outputs",
) -> None:
    """Run full evaluation and generate reports / plots."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model …")
    model, class_names = load_model_and_classes(model_dir)

    print("Loading test dataset …")
    test_ds = get_test_dataset(data_dir)

    print("Running predictions …")
    y_true, y_pred = collect_predictions(model, test_ds)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    print(f"\nTest Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(report)

    # Save report to file
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"Report saved → {report_path}")

    # Plots
    plot_confusion_matrix(y_true, y_pred, class_names, output_dir)
    plot_per_class_metrics(y_true, y_pred, class_names, output_dir)

    # Training history curves (if available)
    history_path = os.path.join(model_dir, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_training_history(history, output_dir)
    else:
        print("No training history found – skipping training curves.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fruit classifier")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    evaluate(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
    )
