"""
evaluation.py
-------------
Quantitative evaluation of retrieval quality:
    • Precision@K
    • Recall@K
    • Inference time per query
    • Comparison plots: Baseline vs Fine-tuned vs Siamese

Usage:
    python src/evaluation.py
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ── Core Metrics ───────────────────────────────────────────────────────────────

def precision_at_k(retrieved_labels: np.ndarray,
                   query_label: int, k: int) -> float:
    """
    Precision@K = (# relevant in top-K) / K
    Relevant = same class as query.
    """
    top_k = retrieved_labels[:k]
    return float(np.sum(top_k == query_label) / k)


def recall_at_k(retrieved_labels: np.ndarray,
                query_label: int,
                total_relevant: int,
                k: int) -> float:
    """
    Recall@K = (# relevant in top-K) / (total relevant in gallery)
    """
    top_k = retrieved_labels[:k]
    hits  = np.sum(top_k == query_label)
    return float(hits / max(total_relevant, 1))


def mean_precision_at_k(embeddings: np.ndarray,
                         labels: np.ndarray,
                         k: int = 5,
                         n_queries: int = 100) -> float:
    """
    Average Precision@K over n_queries randomly chosen samples.
    """
    n = len(labels)
    query_indices = np.random.choice(n, min(n_queries, n), replace=False)
    scores = []

    for qi in query_indices:
        q_emb   = embeddings[qi]
        q_label = labels[qi]
        # cosine similarity (embeddings are L2-normalised)
        sims    = embeddings @ q_emb
        sims[qi] = -1   # exclude self
        ranked  = labels[np.argsort(sims)[::-1]]
        scores.append(precision_at_k(ranked, q_label, k))

    return float(np.mean(scores))


def mean_recall_at_k(embeddings: np.ndarray,
                      labels: np.ndarray,
                      k: int = 5,
                      n_queries: int = 100) -> float:
    n = len(labels)
    query_indices = np.random.choice(n, min(n_queries, n), replace=False)
    scores = []

    for qi in query_indices:
        q_emb        = embeddings[qi]
        q_label      = labels[qi]
        total_rel    = int(np.sum(labels == q_label)) - 1   # exclude self
        sims         = embeddings @ q_emb
        sims[qi]     = -1
        ranked       = labels[np.argsort(sims)[::-1]]
        scores.append(recall_at_k(ranked, q_label, total_rel, k))

    return float(np.mean(scores))


def measure_inference_time(embeddings: np.ndarray,
                            n_queries: int = 50) -> float:
    """Average retrieval time in milliseconds (pure NumPy cosine search)."""
    times = []
    idx   = np.random.choice(len(embeddings), n_queries, replace=False)
    for qi in idx:
        q = embeddings[qi]
        t0 = time.perf_counter()
        _ = embeddings @ q
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


# ── Load helper ────────────────────────────────────────────────────────────────

def load(split_file: str) -> tuple:
    data = np.load(RESULTS_DIR / split_file, allow_pickle=True)
    return data["embeddings"], data["labels"]


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_comparison(results: dict, k_values: list, save: str = "comparison.png"):
    """
    Bar chart comparing Precision@K and Recall@K across models.

    results = {
        "Baseline":  {"precision": [...], "recall": [...]},
        "Fine-tuned": {...},
        "Siamese":   {...}
    }
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Retrieval Performance Comparison", fontsize=14, fontweight="bold")

    palette = ["#378ADD", "#639922", "#D4537E"]
    x = np.arange(len(k_values))
    width = 0.25

    for i, (model_name, metrics) in enumerate(results.items()):
        offset = (i - 1) * width
        ax1.bar(x + offset, metrics["precision"], width,
                label=model_name, color=palette[i], alpha=0.85)
        ax2.bar(x + offset, metrics["recall"], width,
                label=model_name, color=palette[i], alpha=0.85)

    for ax, title in [(ax1, "Precision@K"), (ax2, "Recall@K")]:
        ax.set_xlabel("K")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in k_values])
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / save
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[evaluation] Plot saved → {out}")
    plt.close()


def plot_embedding_distribution(embeddings: np.ndarray,
                                labels: np.ndarray,
                                class_names: list,
                                title: str = "Embedding Distribution",
                                save: str = "embedding_dist.png"):
    """PCA projection of embeddings coloured by class."""
    from sklearn.decomposition import PCA

    pca   = PCA(n_components=2)
    proj  = pca.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("tab10", n_colors=len(class_names))
    for ci, cls in enumerate(class_names):
        mask = labels == ci
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   label=cls, color=palette[ci], alpha=0.6, s=20)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")

    out = RESULTS_DIR / save
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[evaluation] Distribution plot → {out}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    K_VALUES = [1, 3, 5, 10]
    models_info = [
        ("Baseline",  "embeddings_train.npz"),
        ("Fine-tuned","embeddings_train.npz"),   # replace with fine-tuned npz if available
        ("Siamese",   "embeddings_siamese_train.npz"),
    ]

    results = {}
    print(f"\n{'Model':<14} {'K':<5} {'P@K':>8} {'R@K':>8} {'Infer(ms)':>10}")
    print("─" * 52)

    for model_name, fname in models_info:
        fpath = RESULTS_DIR / fname
        if not fpath.exists():
            print(f"[evaluation] {fname} not found — skipping {model_name}")
            continue

        emb, labels = load(fname)
        precision_scores, recall_scores = [], []

        for k in K_VALUES:
            p = mean_precision_at_k(emb, labels, k=k)
            r = mean_recall_at_k(emb, labels, k=k)
            precision_scores.append(p)
            recall_scores.append(r)
            ms = measure_inference_time(emb) if k == K_VALUES[0] else 0
            if k == K_VALUES[0]:
                infer_ms = ms
            print(f"{model_name:<14} K={k:<3} {p:>8.4f} {r:>8.4f} "
                  f"{'':>4}{infer_ms if k == K_VALUES[0] else '':>6.2f}")

        results[model_name] = {
            "precision": precision_scores,
            "recall":    recall_scores
        }

    if results:
        plot_comparison(results, K_VALUES)

    print("\n✅ Evaluation complete. Results in results/")


if __name__ == "__main__":
    main()
