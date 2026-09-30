# 🍎 Fruit Image Classifier

A deep-learning image classifier that identifies fruit types from photos using **MobileNetV2** transfer learning.

**Target classes:** Apple · Banana · Orange · Mango · Grape

---

## Quick Start

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Apple Silicon note:** `tensorflow-metal` is included in `requirements.txt` for GPU acceleration on M-series Macs. It is installed conditionally and can be ignored on other platforms.

### 2. Prepare the Dataset

Downloads the [Fruit Images Dataset](https://github.com/Horea94/Fruit-Images-Dataset) and organises it into `data/train`, `data/validation`, and `data/test` splits.

```bash
python src/prepare_data.py --data-dir data --seed 42
```

If the download fails (network issues), the script automatically generates small synthetic placeholder images so the rest of the pipeline still runs.

### 3. Train the Model

```bash
python src/train.py --data-dir data --model-dir models --epochs 10 --batch-size 32 --lr 1e-3
```

Training runs in two phases:
1. **Classification head** – MobileNetV2 base frozen, only the new dense layers train.
2. **Fine-tuning** – The last 20 base layers are unfrozen and trained at a lower learning rate.

The best checkpoint is saved to `models/best_model.keras`.

### 4. Evaluate

```bash
python src/evaluate.py --data-dir data --model-dir models --output-dir outputs
```

Generates:
- `outputs/classification_report.txt` – accuracy, precision, recall, F1 per class
- `outputs/confusion_matrix.png`
- `outputs/training_curves.png`

### 5. Predict a Single Image

```bash
python src/predict.py path/to/fruit.jpg --model-dir models
```

Example output:

```
Predicted fruit: Apple
Confidence: 94.2%

Class probabilities:
     apple:  94.20%  ████████████████████████████
    banana:   2.10%  █
    orange:   1.80%
     mango:   1.50%
     grape:   0.40%
```

### 6. Streamlit Demo (Optional)

```bash
streamlit run app.py
```

Upload a fruit image in the browser and get instant predictions with a probability chart.

---

## Project Structure

```
fruit-classifier/
├── data/                  # Dataset (auto-generated, git-ignored)
├── models/                # Saved checkpoints (git-ignored)
├── outputs/               # Evaluation plots & reports
├── notebooks/             # (optional) Jupyter experiments
├── src/
│   ├── prepare_data.py    # Dataset download & split
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Metrics & plots
│   └── predict.py         # Single-image inference
├── app.py                 # Streamlit demo UI
├── requirements.txt
├── AGENTS.md
└── README.md
```

## Technical Details

| Component | Choice |
|-----------|--------|
| Base model | MobileNetV2 (ImageNet) |
| Input size | 224 × 224 |
| Optimizer | Adam |
| Loss | Categorical cross-entropy |
| Augmentation | Random flip, rotation, zoom, translation |
| Callbacks | EarlyStopping, ModelCheckpoint |

## Hardware

Designed to run locally on an **Apple Silicon Mac** without requiring an NVIDIA GPU. CPU training works; TensorFlow Metal acceleration is used when available.

---

## License

MIT
