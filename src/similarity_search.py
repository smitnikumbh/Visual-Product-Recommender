"""
similarity_search.py
--------------------
Cosine-similarity–based image retrieval.
Given a query image, returns the top-K most similar images
from the pre-computed embedding index.

Usage (standalone):
    python src/similarity_search.py --query path/to/img.jpg --k 5
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from feature_extractor import (
    build_feature_model,
    load_and_preprocess,
    load_embeddings,
)

RESULTS_DIR = Path("results")


class SimilaritySearcher:
    """
    Wraps the embedding index and performs top-K cosine retrieval.

    Attributes
    ----------
    embeddings  : (N, D) float32  – L2-normalised gallery embeddings
    labels      : (N,)  int       – class ids
    paths       : list[str]       – image file paths
    class_names : list[str]
    model       : Keras Model     – feature extractor (lazy-loaded)
    """

    def __init__(self, split: str = "train"):
        self.embeddings, self.labels, self.paths, self.class_names = \
            load_embeddings(split)
        self._model = None
        print(f"[searcher] Index loaded: {len(self.paths)} images, "
              f"{len(self.class_names)} classes")

    @property
    def model(self):
        if self._model is None:
            self._model = build_feature_model()
        return self._model

    def embed_query(self, img_path: str) -> np.ndarray:
        """Extract and L2-normalise a single query image."""
        arr = load_and_preprocess(img_path)
        emb = self.model.predict(arr, verbose=0)          # (1, 2048)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.squeeze()                               # (2048,)

    def retrieve(self, query_emb: np.ndarray, k: int = 5) -> list[dict]:
        """
        Cosine similarity retrieval (dot product on L2-normalised vectors).

        Returns list of k dicts:
          { rank, path, label, class_name, score (cosine sim) }
        """
        scores = self.embeddings @ query_emb               # (N,)
        top_idx = np.argsort(scores)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_idx, 1):
            results.append({
                "rank":       rank,
                "path":       self.paths[idx],
                "label":      int(self.labels[idx]),
                "class_name": self.class_names[int(self.labels[idx])],
                "score":      float(scores[idx]),
            })
        return results

    def search(self, img_path: str, k: int = 5) -> list[dict]:
        """End-to-end: path → top-K results."""
        q_emb = self.embed_query(img_path)
        return self.retrieve(q_emb, k=k)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Explainable similarity: 1 − cosine_similarity."""
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(1.0 - sim)


def visualise_results(query_path: str, results: list[dict],
                      save_path: str | None = None):
    """
    Plot query image + top-K retrieved images in a grid.
    """
    k = len(results)
    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 4))
    fig.patch.set_facecolor("#f8f8f8")

    # Query
    query_img = Image.open(query_path).convert("RGB")
    axes[0].imshow(query_img)
    axes[0].set_title("Query", fontsize=10, fontweight="bold", color="#333")
    axes[0].axis("off")

    for i, res in enumerate(results, 1):
        try:
            img = Image.open(res["path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(200, 200, 200))
        axes[i].imshow(img)
        axes[i].set_title(
            f"#{res['rank']} {res['class_name']}\ncos-sim: {res['score']:.3f}",
            fontsize=8, color="#555"
        )
        axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[similarity_search] Saved → {save_path}")
    else:
        plt.savefig(RESULTS_DIR / "retrieval_result.png", dpi=150,
                    bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Path to query image")
    parser.add_argument("--k",     type=int, default=5,  help="Top-K results")
    parser.add_argument("--split", default="train",      help="Index split")
    args = parser.parse_args()

    searcher = SimilaritySearcher(split=args.split)
    results  = searcher.search(args.query, k=args.k)

    print(f"\n{'Rank':<6} {'Class':<20} {'Score':>8}  Path")
    print("─" * 70)
    for r in results:
        print(f"{r['rank']:<6} {r['class_name']:<20} {r['score']:>8.4f}  {r['path']}")

    visualise_results(args.query, results)


if __name__ == "__main__":
    main()
