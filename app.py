"""
Streamlit demo UI for the Fruit Image Classifier.

Launch:
    streamlit run app.py
"""

import json
import os

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

IMG_SIZE = (224, 224)
MODEL_DIR = "models"

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍎 Fruit Classifier",
    page_icon="🍇",
    layout="centered",
)


@st.cache_resource
def load_model():
    """Load the trained Keras model (cached across reruns)."""
    model_path = os.path.join(MODEL_DIR, "fruit_classifier.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_DIR, "best_model.keras")
    return keras.models.load_model(model_path)


@st.cache_data
def load_class_names():
    """Load class names from the saved JSON."""
    with open(os.path.join(MODEL_DIR, "class_names.json")) as f:
        return json.load(f)


FRUIT_EMOJI = {
    "apple": "🍎",
    "banana": "🍌",
    "orange": "🍊",
    "mango": "🥭",
    "grape": "🍇",
}


def preprocess(image: Image.Image) -> np.ndarray:
    """Resize and prepare the image for inference."""
    img = image.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


# ── UI ────────────────────────────────────────────────────────────────────
st.title("🍎 Fruit Image Classifier")
st.markdown(
    "Upload a photo of a fruit and the model will predict what it is.  \n"
    "Powered by **MobileNetV2** with transfer learning."
)

uploaded = st.file_uploader(
    "Choose an image …", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Classifying …"):
        model = load_model()
        class_names = load_class_names()

        img_array = preprocess(image)
        preds = model.predict(img_array, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx])
        predicted = class_names[top_idx]

    emoji = FRUIT_EMOJI.get(predicted, "🍽️")
    st.success(f"**{emoji} {predicted.capitalize()}** — {confidence:.1%} confidence")

    # Show all class probabilities as a bar chart
    st.subheader("Class Probabilities")
    prob_dict = {
        f"{FRUIT_EMOJI.get(c, '')} {c.capitalize()}": float(preds[i])
        for i, c in enumerate(class_names)
    }
    st.bar_chart(prob_dict)

    # Detailed table
    with st.expander("Raw probabilities"):
        for name, prob in sorted(prob_dict.items(), key=lambda x: -x[1]):
            st.write(f"**{name}**: {prob:.4%}")

st.markdown("---")
st.caption("Fruit Classifier · MobileNetV2 · TensorFlow / Keras")
