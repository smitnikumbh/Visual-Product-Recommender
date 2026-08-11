"""
dataset_prep.py
---------------
Downloads the Fashion Product Images dataset from Kaggle,
creates a balanced subset of 5–8 categories with ~200–300 images each,
and organises them into train/val splits.

Usage:
    python src/dataset_prep.py
"""

import os
import shutil
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
RAW_DATA_DIR   = Path("data/fashion")          # where you unzipped the Kaggle data
SUBSET_DIR     = Path("data/subset")           # output: organised subset
STYLES_CSV     = RAW_DATA_DIR / "styles.csv"
IMAGES_DIR     = RAW_DATA_DIR / "images"

TARGET_CATEGORIES = [
    "Tshirts", "Shirts", "Casual Shoes", "Watches",
    "Sports Shoes", "Kurtas", "Handbags", "Dresses"
]

IMAGES_PER_CLASS = 250   # ~200–300 as specified
TRAIN_SPLIT      = 0.8
RANDOM_SEED      = 42


def load_metadata() -> pd.DataFrame:
    """Load styles.csv and return cleaned dataframe."""
    df = pd.read_csv(STYLES_CSV, on_bad_lines="skip")
    df = df[["id", "articleType", "baseColour", "season", "usage"]].dropna()
    df["image_path"] = df["id"].astype(str).apply(
        lambda x: IMAGES_DIR / f"{x}.jpg"
    )
    # keep only rows whose image actually exists
    df = df[df["image_path"].apply(lambda p: p.exists())]
    print(f"[dataset_prep] Total valid rows: {len(df)}")
    return df


def create_subset(df: pd.DataFrame) -> dict:
    """
    Sample IMAGES_PER_CLASS images from each TARGET_CATEGORY.
    Returns dict: {category: [list of src paths]}
    """
    random.seed(RANDOM_SEED)
    subset = {}

    for cat in TARGET_CATEGORIES:
        cat_df = df[df["articleType"] == cat]
        n = min(IMAGES_PER_CLASS, len(cat_df))
        if n < 50:
            print(f"[dataset_prep] ⚠  {cat}: only {n} images — skipping")
            continue
        sampled = cat_df.sample(n, random_state=RANDOM_SEED)["image_path"].tolist()
        subset[cat] = sampled
        print(f"[dataset_prep] {cat}: {n} images selected")

    return subset


def copy_to_subset_dir(subset: dict):
    """Copy sampled images into data/subset/{train|val}/{category}/."""
    if SUBSET_DIR.exists():
        shutil.rmtree(SUBSET_DIR)

    for split in ("train", "val"):
        for cat in subset:
            (SUBSET_DIR / split / cat).mkdir(parents=True, exist_ok=True)

    for cat, paths in subset.items():
        random.shuffle(paths)
        split_idx = int(len(paths) * TRAIN_SPLIT)
        splits = {"train": paths[:split_idx], "val": paths[split_idx:]}

        for split, img_paths in splits.items():
            for src in tqdm(img_paths, desc=f"Copying {cat}/{split}", leave=False):
                dst = SUBSET_DIR / split / cat / Path(src).name
                shutil.copy2(src, dst)

    # summary
    for split in ("train", "val"):
        total = sum(
            len(list((SUBSET_DIR / split / cat).iterdir()))
            for cat in subset
        )
        print(f"[dataset_prep] {split}: {total} images")


def get_class_names() -> list:
    """Return list of category names from subset directory."""
    return sorted([d.name for d in (SUBSET_DIR / "train").iterdir() if d.is_dir()])


def main():
    if not STYLES_CSV.exists():
        print(
            "styles.csv not found!\n"
            "Download the dataset from:\n"
            "  https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset\n"
            "Unzip into data/fashion/"
        )
        return

    print("[dataset_prep] Loading metadata …")
    df = load_metadata()

    print("[dataset_prep] Creating subset …")
    subset = create_subset(df)

    print("[dataset_prep] Copying files …")
    copy_to_subset_dir(subset)

    print("\n✅ Subset created at:", SUBSET_DIR.resolve())
    print("   Classes:", list(subset.keys()))


if __name__ == "__main__":
    main()
