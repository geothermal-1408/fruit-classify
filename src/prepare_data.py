from __future__ import annotations

"""
Dataset preparation script for the Fruit Image Classifier.

Downloads a small fruit-image dataset and organises it into
train / validation / test splits following the structure expected
by the training pipeline.

Usage:
    python src/prepare_data.py [--data-dir data] [--seed 42]
"""

import argparse
import os
import pathlib
import random
import shutil
import urllib.request
import zipfile

# ── Tiny built-in dataset generator ───────────────────────────────────────
# We use TensorFlow/Keras image_dataset_from_directory later, so we need
# real images on disk.  The script will attempt to download a curated
# fruit-images subset.  If that fails (network issues), it falls back to
# generating small synthetic placeholder images with Pillow so the rest of
# the pipeline still works.

CLASSES = ["apple", "banana", "orange", "mango", "grape"]

# FIDS30 is a real-world fruit dataset with complex backgrounds and lighting.
DATASET_URL = "https://data.vicos.si/datasets/FIDS30/FIDS30.zip"

# Mapping from FIDS30 dataset folder names to our target classes
FOLDER_MAP = {
    "apple": ["apples"],
    "banana": ["bananas"],
    "orange": ["oranges"],
    "pineapple": ["pineapples"],
    "grape": ["grapes"],
}

IMAGES_PER_CLASS = 600  # cap per split source
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42


def _create_dirs(data_dir: str) -> None:
    """Create train/validation/test subdirectories for each class."""
    for split in ("train", "validation", "test"):
        for cls in CLASSES:
            os.makedirs(os.path.join(data_dir, split, cls), exist_ok=True)


def _download_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Callback for urlretrieve that prints download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  [{bar}] {pct:5.1f}%  {mb_down:.1f} / {mb_total:.1f} MB", end="", flush=True)
    else:
        mb_down = downloaded / (1024 * 1024)
        print(f"\r  Downloaded {mb_down:.1f} MB …", end="", flush=True)


def _download_and_extract(dest: str) -> str | None:
    """Download the fruit dataset ZIP and extract; return extracted root or None."""
    zip_path = os.path.join(dest, "fruits.zip")
    print(f"Downloading dataset from {DATASET_URL}")
    print(f"  Saving to: {zip_path}")
    try:
        urllib.request.urlretrieve(DATASET_URL, zip_path, reporthook=_download_progress)
        print()  # newline after progress bar
    except KeyboardInterrupt:
        print("\n\n  ⚠ Download interrupted by user.")
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"  Cleaned up partial file: {zip_path}")
        return None
    except Exception as exc:
        print(f"\n  ⚠ Download failed: {exc}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"  Cleaned up partial file: {zip_path}")
        return None

    # Verify the ZIP is valid (not truncated)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                print(f"  ⚠ Corrupt ZIP (bad file: {bad}). Deleting.")
                os.remove(zip_path)
                return None
    except zipfile.BadZipFile:
        print("  ⚠ Downloaded file is not a valid ZIP (incomplete download?). Deleting.")
        os.remove(zip_path)
        return None

    file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  Download complete: {file_size_mb:.1f} MB")

    print("Extracting …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    os.remove(zip_path)
    print("  Extraction complete, ZIP deleted.")

    # Find the extracted root (the FIDS30 folder)
    # Skip our own train/validation/test dirs
    for entry in sorted(os.listdir(dest)):
        candidate = os.path.join(dest, entry)
        if not os.path.isdir(candidate):
            continue
        if entry in ("train", "validation", "test"):
            continue
        # Check if this looks like the FIDS30 dataset (has fruit subfolders)
        has_apples = os.path.isdir(os.path.join(candidate, "apples"))
        has_bananas = os.path.isdir(os.path.join(candidate, "bananas"))
        if has_apples or has_bananas:
            print(f"  Found dataset root: {candidate}")
            return candidate
    return None

def _collect_images(extracted_root: str) -> dict[str, list[str]]:
    """Walk the dataset and collect image paths per target class."""
    collected: dict[str, list[str]] = {cls: [] for cls in CLASSES}

    for cls, folder_names in FOLDER_MAP.items():
        for folder_name in folder_names:
            folder_path = os.path.join(extracted_root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    collected[cls].append(os.path.join(folder_path, fname))
    return collected


def _split_and_copy(
    images: dict[str, list[str]], data_dir: str, seed: int
) -> None:
    """Randomly split collected images into train/val/test and copy them."""
    rng = random.Random(seed)
    for cls, paths in images.items():
        rng.shuffle(paths)
        paths = paths[: IMAGES_PER_CLASS]  # cap
        n = len(paths)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        splits = {
            "train": paths[:n_train],
            "validation": paths[n_train : n_train + n_val],
            "test": paths[n_train + n_val :],
        }
        for split, split_paths in splits.items():
            dest_dir = os.path.join(data_dir, split, cls)
            for i, src in enumerate(split_paths):
                ext = pathlib.Path(src).suffix
                dst = os.path.join(dest_dir, f"{cls}_{i:04d}{ext}")
                shutil.copy2(src, dst)
        print(
            f"  {cls}: train={len(splits['train'])}, "
            f"val={len(splits['validation'])}, test={len(splits['test'])}"
        )


def _generate_synthetic(data_dir: str, seed: int) -> None:
    """Generate tiny coloured placeholder images so the pipeline can run."""
    from PIL import Image  # local import – only needed for fallback

    COLOR_MAP = {
        "apple": (200, 30, 30),
        "banana": (240, 220, 60),
        "orange": (255, 165, 0),
        "mango": (255, 200, 50),
        "grape": (120, 50, 170),
    }
    rng = random.Random(seed)
    per_class = 120
    for cls in CLASSES:
        base = COLOR_MAP[cls]
        all_imgs: list[str] = []
        for i in range(per_class):
            img = Image.new("RGB", (224, 224))
            pixels = img.load()
            for x in range(224):
                for y in range(224):
                    noise = tuple(
                        max(0, min(255, c + rng.randint(-30, 30))) for c in base
                    )
                    pixels[x, y] = noise  # type: ignore[index]
            tmp_path = os.path.join(data_dir, f"_tmp_{cls}_{i}.jpg")
            img.save(tmp_path)
            all_imgs.append(tmp_path)

        rng.shuffle(all_imgs)
        n_train = int(per_class * TRAIN_RATIO)
        n_val = int(per_class * VAL_RATIO)
        splits = {
            "train": all_imgs[:n_train],
            "validation": all_imgs[n_train : n_train + n_val],
            "test": all_imgs[n_train + n_val :],
        }
        for split, paths in splits.items():
            dest_dir = os.path.join(data_dir, split, cls)
            for j, src in enumerate(paths):
                dst = os.path.join(dest_dir, f"{cls}_{j:04d}.jpg")
                shutil.move(src, dst)
        print(
            f"  {cls} (synthetic): train={len(splits['train'])}, "
            f"val={len(splits['validation'])}, test={len(splits['test'])}"
        )


def _count_images(directory: str) -> int:
    """Count image files recursively inside a directory."""
    count = 0
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                count += 1
    return count


def prepare(data_dir: str = "data", seed: int = SEED) -> None:
    """Main entry-point: download (or synthesise) and organise the dataset."""
    data_dir = os.path.abspath(data_dir)

    # Check if dataset already has actual images (not just empty dirs)
    train_dir = os.path.join(data_dir, "train")
    if os.path.isdir(train_dir) and _count_images(train_dir) > 0:
        n = _count_images(train_dir)
        print(f"Dataset already exists at {data_dir} ({n} training images) – skipping.")
        return

    # Clean up any leftover partial ZIP from a previous interrupted run
    stale_zip = os.path.join(data_dir, "fruits.zip")
    if os.path.exists(stale_zip):
        size_mb = os.path.getsize(stale_zip) / (1024 * 1024)
        print(f"Found leftover fruits.zip ({size_mb:.1f} MB) from a previous run – deleting.")
        os.remove(stale_zip)

    _create_dirs(data_dir)

    # Check if the dataset was already extracted (e.g. previous run downloaded
    # successfully but failed during splitting)
    download_dir = data_dir
    extracted = None
    for entry in sorted(os.listdir(data_dir)):
        candidate = os.path.join(data_dir, entry)
        if not os.path.isdir(candidate) or entry in ("train", "validation", "test"):
            continue
        if os.path.isdir(os.path.join(candidate, "apples")) or \
           os.path.isdir(os.path.join(candidate, "bananas")):
            print(f"Found already-extracted dataset: {candidate}")
            extracted = candidate
            break

    # Download only if not already extracted
    if extracted is None:
        extracted = _download_and_extract(download_dir)
    if extracted:
        images = _collect_images(extracted)
        total = sum(len(v) for v in images.values())
        if total > 0:
            print(f"Collected {total} images from dataset. Splitting …")
            _split_and_copy(images, data_dir, seed)
            # Clean up the extracted source folder
            shutil.rmtree(extracted, ignore_errors=True)
            return
        # Clean up if no images found
        shutil.rmtree(extracted, ignore_errors=True)

    # Fallback: synthetic images
    print("Generating synthetic placeholder images …")
    _generate_synthetic(data_dir, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare fruit classification dataset")
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Root data directory"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()
    prepare(data_dir=args.data_dir, seed=args.seed)
