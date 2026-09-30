"""
Single-image prediction script for the Fruit Image Classifier.

Usage:
    python src/predict.py path/to/image.jpg [--model-dir models]
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = (224, 224)


def load_model_and_classes(model_dir: str):
    """Load the trained model and class names."""
    model_path = os.path.join(model_dir, "fruit_classifier.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "best_model.keras")
    if not os.path.exists(model_path):
        print(f"Error: No model found in {model_dir}")
        sys.exit(1)

    model = keras.models.load_model(model_path)
    class_names_path = os.path.join(model_dir, "class_names.json")
    with open(class_names_path) as f:
        class_names = json.load(f)
    return model, class_names


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess a single image for MobileNetV2."""
    if not os.path.exists(image_path):
        print(f"Error: Image not found – {image_path}")
        sys.exit(1)

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        print(f"Error: Cannot open image – {exc}")
        sys.exit(1)

    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    # Add batch dimension
    return np.expand_dims(arr, axis=0)


def predict(image_path: str, model_dir: str = "models") -> dict:
    """Run prediction on a single image and return results dict."""
    model, class_names = load_model_and_classes(model_dir)
    img = preprocess_image(image_path)

    preds = model.predict(img, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])

    result = {
        "predicted_fruit": class_names[top_idx],
        "confidence": confidence,
        "probabilities": {
            class_names[i]: float(preds[i]) for i in range(len(class_names))
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict fruit from an image")
    parser.add_argument("image", type=str, help="Path to the image file")
    parser.add_argument("--model-dir", type=str, default="models")
    args = parser.parse_args()

    result = predict(args.image, args.model_dir)

    print(f"\nPredicted fruit: {result['predicted_fruit']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print("\nClass probabilities:")
    for name, prob in sorted(
        result["probabilities"].items(), key=lambda x: -x[1]
    ):
        bar = "█" * int(prob * 30)
        print(f"  {name:>8s}: {prob:6.2%}  {bar}")


if __name__ == "__main__":
    main()
