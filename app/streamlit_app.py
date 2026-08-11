"""
streamlit_app.py
----------------
Interactive UI for the Visual Product Recommendation System.

Launch:
    streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import time
import io

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Visual Product Recommender",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(90deg, #5533aa, #d4537e);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #f8f8fb; border: 1px solid #e0dff0;
        border-radius: 12px; padding: 1rem; text-align: center;
    }
    .result-card {
        border: 1px solid #e8e8e8; border-radius: 10px;
        padding: 0.5rem; text-align: center; background: white;
    }
    .score-bar { height: 6px; border-radius: 3px; background: #eeee; }
    .score-fill { height: 100%; border-radius: 3px;
                  background: linear-gradient(90deg, #5533aa, #d4537e); }
    .stButton button { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading feature extractor …")
def get_model(model_type: str):
    """Load the appropriate embedding model."""
    from tensorflow.keras.models import load_model as keras_load
    from feature_extractor import build_feature_model

    MODELS_DIR = Path("models")

    if model_type == "Siamese" and (MODELS_DIR / "siamese_embedding_net.h5").exists():
        model = keras_load(MODELS_DIR / "siamese_embedding_net.h5", compile=False)
        st.sidebar.success("✅ Siamese model loaded")
    elif model_type == "Fine-tuned" and (MODELS_DIR / "feature_model_finetuned.h5").exists():
        model = keras_load(MODELS_DIR / "feature_model_finetuned.h5", compile=False)
        st.sidebar.success("✅ Fine-tuned model loaded")
    else:
        model = build_feature_model()
        st.sidebar.info("ℹ️  Using baseline ResNet50")
    return model


@st.cache_data(show_spinner="Loading embedding index …")
def load_index(model_type: str) -> tuple:
    """Load pre-computed embeddings from disk."""
    RESULTS_DIR = Path("results")

    if model_type == "Siamese":
        fname = RESULTS_DIR / "embeddings_siamese_train.npz"
    else:
        fname = RESULTS_DIR / "embeddings_train.npz"

    if not fname.exists():
        return None, None, None, None

    data = np.load(fname, allow_pickle=True)
    return (
        data["embeddings"],
        data["labels"],
        data["paths"].tolist(),
        data["class_names"].tolist()
    )


# ── Inference helpers ──────────────────────────────────────────────────────────

def embed_uploaded(model, img: Image.Image) -> np.ndarray:
    """Preprocess uploaded PIL image → L2-normalised embedding."""
    from tensorflow.keras.applications.resnet50 import preprocess_input
    import numpy as np

    img_resized = img.convert("RGB").resize((224, 224))
    arr = np.array(img_resized, dtype=np.float32)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)
    emb = model.predict(arr, verbose=0)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb.squeeze()


def retrieve_top_k(query_emb: np.ndarray,
                   embeddings: np.ndarray,
                   labels: np.ndarray,
                   paths: list,
                   class_names: list,
                   k: int = 5) -> list[dict]:
    scores   = embeddings @ query_emb
    top_idx  = np.argsort(scores)[::-1][:k]
    results  = []
    for rank, idx in enumerate(top_idx, 1):
        results.append({
            "rank":       rank,
            "path":       paths[idx],
            "label":      int(labels[idx]),
            "class_name": class_names[int(labels[idx])],
            "score":      float(scores[idx]),
        })
    return results


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    model_type = st.selectbox(
        "Embedding model",
        ["Baseline (ResNet50)", "Fine-tuned", "Siamese"],
        index=2,
        help="Siamese gives best retrieval quality"
    )
    model_key = model_type.split(" ")[0]   # "Baseline" | "Fine-tuned" | "Siamese"

    k_results = st.slider("Top-K results", 1, 12, 6)

    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown(
        "Visual similarity search using deep learning. "
        "Upload any fashion product image to find visually similar items."
    )

    st.markdown("---")
    st.markdown("**Model legend**")
    st.markdown("🔵 Baseline — cosine on raw ResNet50")
    st.markdown("🟢 Fine-tuned — last 30 layers re-trained")
    st.markdown("🟣 Siamese — triplet loss embedding space ✦ best")


# ── Main Page ──────────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">👗 Visual Product Recommender</p>',
            unsafe_allow_html=True)
st.markdown(
    "Upload a fashion product image — the system retrieves the most visually "
    "similar products using deep learning embeddings.",
    help="Powered by ResNet50 + Siamese Network"
)

col_upload, col_results = st.columns([1, 2.5], gap="large")

with col_upload:
    st.markdown("### 📤 Upload image")
    uploaded = st.file_uploader(
        "Drag & drop or browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Query image", use_container_width=True)
        st.markdown(f"**Size:** {img.size[0]}×{img.size[1]} px")

        search_btn = st.button("🔍 Find similar products", type="primary",
                               use_container_width=True)
    else:
        st.info("Upload an image to start searching.")
        search_btn = False

with col_results:
    if uploaded and search_btn:
        with st.spinner("Extracting features …"):
            t0    = time.perf_counter()
            model = get_model(model_key)
            emb, labels, paths, class_names = load_index(model_key)
            infer_start = time.perf_counter()
            query_emb   = embed_uploaded(model, img)
            infer_ms    = (time.perf_counter() - infer_start) * 1000

        if emb is None:
            st.warning(
                "⚠️ No embedding index found. "
                "Run `python src/feature_extractor.py` first to build the index."
            )
        else:
            t_search = time.perf_counter()
            results  = retrieve_top_k(
                query_emb, emb, labels, paths, class_names, k=k_results
            )
            search_ms = (time.perf_counter() - t_search) * 1000

            # ── Metrics row ───────────────────────────────────────────────────
            st.markdown("### 📈 Performance")
            m1, m2, m3 = st.columns(3)
            m1.metric("Model", model_type.split(" ")[0])
            m2.metric("Embed time", f"{infer_ms:.1f} ms")
            m3.metric("Search time", f"{search_ms:.2f} ms")

            # ── Results grid ─────────────────────────────────────────────────
            st.markdown(f"### 🎯 Top-{k_results} similar products")
            cols_per_row = 3
            for row_start in range(0, len(results), cols_per_row):
                row_results = results[row_start: row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, res in zip(cols, row_results):
                    with col:
                        try:
                            r_img = Image.open(res["path"]).convert("RGB")
                        except Exception:
                            r_img = Image.new("RGB", (224, 224), (230, 230, 230))

                        st.image(r_img, use_container_width=True)
                        score_pct = int(res["score"] * 100)
                        st.markdown(
                            f"**#{res['rank']} · {res['class_name']}**  \n"
                            f"Similarity: `{res['score']:.3f}`  \n"
                            f"<div class='score-bar'>"
                            f"<div class='score-fill' style='width:{score_pct}%'></div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

            # ── Cosine distance explanation ───────────────────────────────────
            with st.expander("📐 Explainable similarity — cosine distance"):
                st.markdown(
                    "**Cosine similarity** measures the angle between two embedding "
                    "vectors in high-dimensional space. A score of **1.0** = identical, "
                    "**0.0** = orthogonal (unrelated)."
                )
                st.markdown("| Rank | Class | Cosine Sim | Distance |")
                st.markdown("|------|-------|-----------|---------|")
                for r in results:
                    dist = 1.0 - r["score"]
                    st.markdown(
                        f"| #{r['rank']} | {r['class_name']} "
                        f"| {r['score']:.4f} | {dist:.4f} |"
                    )

    elif not uploaded:
        st.markdown("### 🎯 Results will appear here")
        st.markdown("Upload an image on the left to search.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>Visual Product Recommendation System · "
    "ResNet50 + Siamese Network · MCA Project · MIT-WPU</small>",
    unsafe_allow_html=True
)
