"""
utils.py
--------
Shared utility functions used across the project.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os


def load_image_pil(path: str, size: tuple = (224, 224)) -> Image.Image:
    """Load and resize image using PIL."""
    return Image.open(path).convert("RGB").resize(size)


def image_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    """Convert PIL Image to bytes (for Streamlit display)."""
    import io
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def batch_cosine_similarity(query: np.ndarray,
                             gallery: np.ndarray) -> np.ndarray:
    """
    Vectorised cosine similarity of query (D,) against gallery (N, D).
    Both assumed to be L2-normalised → dot product suffices.
    """
    return gallery @ query


def plot_retrieval_grid(query_path: str,
                         results: list[dict],
                         save_path: str | None = None,
                         title: str = "Visual Retrieval Results") -> plt.Figure:
    """
    Plot query image + top-K retrieved images in a single row.
    """
    k   = len(results)
    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 4))
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.patch.set_facecolor("#f5f5f5")

    # Query image
    q_img = Image.open(query_path).convert("RGB")
    axes[0].imshow(q_img)
    axes[0].set_title("Query", fontsize=9, fontweight="bold")
    axes[0].axis("off")
    for spine in axes[0].spines.values():
        spine.set_edgecolor("#5533aa")
        spine.set_linewidth(2)

    palette = plt.cm.get_cmap("RdYlGn", k + 1)
    for i, res in enumerate(results, 1):
        try:
            img = Image.open(res["path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (220, 220, 220))

        axes[i].imshow(img)
        score = res.get("score", 0.0)
        axes[i].set_title(
            f"#{res['rank']}  {res.get('class_name', '')}",
            fontsize=8
        )
        axes[i].set_xlabel(f"cos-sim: {score:.3f}", fontsize=7, color="#555")
        axes[i].tick_params(left=False, bottom=False,
                            labelleft=False, labelbottom=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def get_class_distribution(class_index: dict) -> dict:
    """Return {class_name: count} from class_index."""
    return {cls: len(paths) for cls, paths in class_index.items()}


def pca_2d(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Reduce embeddings to 2D via PCA (for visualisation)."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(embeddings)
