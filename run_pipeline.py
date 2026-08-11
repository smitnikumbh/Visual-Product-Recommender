"""
run_pipeline.py
---------------
ONE-CLICK SETUP — Run this after unzipping the project.
Downloads Fashion dataset via Kaggle API, runs full pipeline,
generates embeddings + results.

Usage:
    pip install -r requirements.txt
    python run_pipeline.py
"""

import os
import sys
import subprocess
from pathlib import Path

def run(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        sys.exit(1)

def check_kaggle():
    """Check kaggle.json exists"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("""
╔══════════════════════════════════════════════════════════╗
║         KAGGLE API KEY SETUP (one-time only)            ║
╠══════════════════════════════════════════════════════════╣
║  1. Go to: https://www.kaggle.com/settings/account      ║
║  2. Scroll to API section → "Create New Token"          ║
║  3. This downloads kaggle.json                          ║
║  4. Move it to:                                         ║
║       Windows: C:\\Users\\YOU\\.kaggle\\kaggle.json       ║
║       Mac/Linux: ~/.kaggle/kaggle.json                  ║
║  5. Run this script again                               ║
╚══════════════════════════════════════════════════════════╝
""")
        sys.exit(1)
    print("✅ Kaggle API key found")

def download_dataset():
    data_dir = Path("data/fashion")
    if (data_dir / "styles.csv").exists():
        print("✅ Dataset already downloaded, skipping...")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    print("📥 Downloading Fashion Product Images dataset (~25GB)...")
    print("   This will take a while depending on your connection.")

    run(
        "kaggle datasets download -d paramaggarwal/fashion-product-images-dataset "
        "-p data/fashion/ --unzip",
        "Downloading from Kaggle"
    )
    print("✅ Dataset downloaded!")

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║     Visual Product Recommendation System — Setup        ║
╚══════════════════════════════════════════════════════════╝
""")

    # 1. Check Kaggle API
    check_kaggle()

    # 2. Download dataset
    download_dataset()

    # 3. Prepare subset
    run("python src/dataset_prep.py", "Step 1/4: Creating dataset subset (5-8 categories, ~250 imgs each)")

    # 4. Extract baseline features
    run("python src/feature_extractor.py", "Step 2/4: Extracting ResNet50 embeddings")

    # 5. Train Siamese network
    print("""
╔══════════════════════════════════════════════════════════╗
║  Step 3/4: Siamese Network Training                     ║
║  This takes ~20-40 mins on CPU, ~5 mins on GPU          ║
║  Press Ctrl+C to skip and use baseline only             ║
╚══════════════════════════════════════════════════════════╝
""")
    try:
        run("python src/siamese_network.py", "Training Siamese Network")
    except KeyboardInterrupt:
        print("\n⏭️  Siamese training skipped — baseline still works!")

    # 6. Evaluate
    run("python src/evaluation.py", "Step 4/4: Evaluating Precision@K and Recall@K")

    print("""
╔══════════════════════════════════════════════════════════╗
║                   ✅ SETUP COMPLETE!                    ║
║                                                         ║
║  Launch the app:                                        ║
║      streamlit run app/streamlit_app.py                 ║
║                                                         ║
║  Results saved in: results/                             ║
╚══════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    main()
