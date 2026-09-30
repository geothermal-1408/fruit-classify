"""
Training pipeline for the Fruit Image Classifier.

Builds a MobileNetV2-based transfer-learning model, trains on the prepared
dataset, and saves the best checkpoint.

Usage:
    python src/train.py [--data-dir data] [--model-dir models]
                        [--epochs 10] [--batch-size 32] [--lr 1e-3]
                        [--seed 42] [--fine-tune-layers 20]
"""

import argparse
import os
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks  # type: ignore[attr-defined]

# ── Defaults ──────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
SEED = 42
FINE_TUNE_LAYERS = 40  # number of layers to unfreeze during fine-tuning (0 = skip)

CLASSES = ["apple", "banana", "grape", "mango", "orange"]


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_datasets(
    data_dir: str, batch_size: int, seed: int
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Load train / val / test datasets from the directory structure."""
    train_ds = keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        seed=seed,
        shuffle=True,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "validation"),
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        seed=seed,
        shuffle=False,
    )
    test_ds = keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "test"),
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        seed=seed,
        shuffle=False,
    )
    class_names = sorted(train_ds.class_names)
    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes: int, lr: float) -> tuple:
    """Build a MobileNetV2 transfer-learning model.

    Returns:
        (model, base_model) – the full model and the MobileNetV2 base.
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # freeze all base layers initially

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    # MobileNetV2 preprocessing: scale pixels to [-1, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def get_augmentation_layer() -> keras.Sequential:
    """Return a Keras data-augmentation layer."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomContrast(0.1),
            layers.RandomBrightness(0.1),
        ],
        name="augmentation",
    )


def augment_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Apply data augmentation to a training dataset."""
    aug = get_augmentation_layer()
    return ds.map(lambda x, y: (aug(x, training=True), y),
                  num_parallel_calls=tf.data.AUTOTUNE)


def train(
    data_dir: str = "data",
    model_dir: str = "models",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    seed: int = SEED,
    fine_tune_layers: int = FINE_TUNE_LAYERS,
) -> None:
    """Full training pipeline."""
    set_seeds(seed)
    os.makedirs(model_dir, exist_ok=True)

    # 1. Load data
    print("Loading datasets …")
    train_ds, val_ds, test_ds, class_names = load_datasets(data_dir, batch_size, seed)
    print(f"Classes: {class_names}")

    # Save class names for later inference
    with open(os.path.join(model_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    # 2. Augment training data & prefetch
    train_ds = augment_dataset(train_ds).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # 3. Build model
    print("Building model …")
    model, base_model = build_model(num_classes=len(class_names), lr=lr)
    model.summary()

    # 4. Callbacks
    checkpoint_path = os.path.join(model_dir, "best_model.keras")
    cb = [
        callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    # 5. Train classification head
    print("\n── Phase 1: Training classification head ──")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb,
    )

    # 6. Optional fine-tuning
    if fine_tune_layers > 0:
        print(f"\n── Phase 2: Fine-tuning last {fine_tune_layers} base layers ──")
        base_model.trainable = True
        # Freeze all but the last `fine_tune_layers` layers
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr / 10),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        ft_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=cb,
        )
        # Merge histories
        for k in history.history:
            history.history[k].extend(ft_history.history[k])

    # 7. Save final model
    model.save(os.path.join(model_dir, "fruit_classifier.keras"))
    print(f"\nModel saved to {model_dir}/fruit_classifier.keras")

    # 8. Save training history
    history_path = os.path.join(model_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    print(f"Training history saved to {history_path}")

    # 9. Quick test evaluation
    print("\n── Test set evaluation ──")
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    loss, acc = model.evaluate(test_ds)
    print(f"Test loss: {loss:.4f} | Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fruit classifier")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fine-tune-layers", type=int, default=FINE_TUNE_LAYERS)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        fine_tune_layers=args.fine_tune_layers,
    )
