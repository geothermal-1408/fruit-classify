"""
Evaluation script for the Fruit Image Classifier.

Loads a trained model and computes:
  • Test accuracy, precision, recall, F1-score (per-class & macro)
  • Confusion matrix (saved as PNG)
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
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


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


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Generate and save a confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {path}")


def plot_training_history(history: dict, output_dir: str):
    """Plot training vs validation accuracy and loss."""
    epochs_range = range(1, len(history["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(epochs_range, history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(epochs_range, history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[0].set_title("Accuracy", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(epochs_range, history["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(epochs_range, history["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_title("Loss", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved → {path}")


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
