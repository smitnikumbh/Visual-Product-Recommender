"""
feature_extractor.py
--------------------
Loads pretrained ResNet50 (no classification head),
extracts 2048-dim embeddings for every image in the subset,
and saves them to results/embeddings.npz.

Usage:
    python src/feature_extractor.py
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras import Model

# ── Config ─────────────────────────────────────────────────────────────────────
SUBSET_DIR   = Path("data/subset")
RESULTS_DIR  = Path("results")
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
RESULTS_DIR.mkdir(exist_ok=True)


def build_feature_model(weights: str = "imagenet") -> Model:
    """
    ResNet50 with the top (FC + softmax) removed.
    Output shape: (None, 2048)
    """
    base = ResNet50(weights=weights, include_top=False,
                    input_shape=(*IMG_SIZE, 3), pooling="avg")
    base.trainable = False
    print(f"[feature_extractor] Model output shape: {base.output_shape}")
    return base


def load_and_preprocess(img_path: str) -> np.ndarray:
    """Load single image → normalised numpy array."""
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)   # ImageNet mean/std normalisation


def extract_embeddings(model: Model, split: str = "train") -> tuple:
    """
    Walk subset/{split}/** and extract embeddings for every image.
    Returns:
        embeddings : np.ndarray  shape (N, 2048)
        labels     : np.ndarray  shape (N,)  integer class ids
        paths      : list[str]
        class_names: list[str]
    """
    split_dir = SUBSET_DIR / split
    class_names = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    all_paths, all_labels = [], []
    for cls in class_names:
        cls_dir = split_dir / cls
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in cls_dir.glob(ext):
                all_paths.append(str(p))
                all_labels.append(class_to_idx[cls])

    print(f"[feature_extractor] {split}: {len(all_paths)} images, "
          f"{len(class_names)} classes")

    embeddings = []
    for i in tqdm(range(0, len(all_paths), BATCH_SIZE),
                  desc=f"Extracting {split}"):
        batch_paths = all_paths[i: i + BATCH_SIZE]
        batch = np.vstack([load_and_preprocess(p) for p in batch_paths])
        emb = model.predict(batch, verbose=0)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    # L2-normalise so cosine similarity = dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    return embeddings, np.array(all_labels), all_paths, class_names


def save_embeddings(embeddings, labels, paths, class_names, split: str):
    out = RESULTS_DIR / f"embeddings_{split}.npz"
    np.savez_compressed(
        out,
        embeddings=embeddings,
        labels=labels,
        paths=np.array(paths),
        class_names=np.array(class_names)
    )
    print(f"[feature_extractor] Saved → {out}  shape={embeddings.shape}")


def load_embeddings(split: str = "train") -> tuple:
    """Convenience loader for downstream modules."""
    data = np.load(RESULTS_DIR / f"embeddings_{split}.npz", allow_pickle=True)
    return (
        data["embeddings"],
        data["labels"],
        data["paths"].tolist(),
        data["class_names"].tolist()
    )


def main():
    model = build_feature_model()

    for split in ("train", "val"):
        if not (SUBSET_DIR / split).exists():
            print(f"[feature_extractor] {split} split not found — run dataset_prep.py first")
            continue
        emb, labels, paths, classes = extract_embeddings(model, split)
        save_embeddings(emb, labels, paths, classes, split)

    print("\n✅ Feature extraction complete.")


if __name__ == "__main__":
    main()
