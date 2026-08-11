"""
siamese_network.py
------------------
Siamese Network with Triplet Loss trained on fashion subset.

Architecture:
    Shared CNN Backbone (ResNet50, pretrained)
    → Dense 512 (L2-normalised)  ← embedding space

Triplet Loss:
    L = max(0,  d(anchor, positive) − d(anchor, negative) + margin)
    where d(·,·) is squared Euclidean distance.

Usage:
    python src/siamese_network.py
"""

import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import Model, layers, backend as K
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ── Config ─────────────────────────────────────────────────────────────────────
SUBSET_DIR      = Path("data/subset")
MODELS_DIR      = Path("models")
RESULTS_DIR     = Path("results")
IMG_SIZE        = (224, 224)
EMBEDDING_DIM   = 512
MARGIN          = 0.5
BATCH_SIZE      = 32
EPOCHS          = 20
LEARNING_RATE   = 1e-4
TRIPLETS_PER_CLASS = 50   # anchor-pos-neg triplets sampled per class per epoch
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ── Model ──────────────────────────────────────────────────────────────────────

def build_embedding_network() -> Model:
    """
    Shared-weight tower: ResNet50 backbone → Dense(512) → L2-normalise.
    This single model is reused for all three inputs (anchor, pos, neg).
    """
    base = ResNet50(weights="imagenet", include_top=False,
                    input_shape=(*IMG_SIZE, 3), pooling="avg")
    # Freeze all base layers initially; fine-tuning optional
    for layer in base.layers[:-30]:
        layer.trainable = False

    inp = layers.Input(shape=(*IMG_SIZE, 3), name="image_input")
    x   = base(inp, training=False)
    x   = layers.Dense(EMBEDDING_DIM, activation=None, name="embedding")(x)
    x   = layers.Lambda(lambda t: K.l2_normalize(t, axis=1), name="l2_norm")(x)
    return Model(inputs=inp, outputs=x, name="EmbeddingNet")


def build_siamese_model(emb_net: Model) -> Model:
    """
    Full triplet model:
        inputs  → (anchor, positive, negative) each (224,224,3)
        output  → concatenated [emb_a, emb_p, emb_n]  shape (None, 512*3)
    Loss computed externally via TripletLoss layer below.
    """
    anchor_inp   = layers.Input(shape=(*IMG_SIZE, 3), name="anchor")
    positive_inp = layers.Input(shape=(*IMG_SIZE, 3), name="positive")
    negative_inp = layers.Input(shape=(*IMG_SIZE, 3), name="negative")

    emb_a = emb_net(anchor_inp)
    emb_p = emb_net(positive_inp)
    emb_n = emb_net(negative_inp)

    # Stack all embeddings for the custom loss to use
    merged = layers.concatenate([emb_a, emb_p, emb_n], axis=1, name="triplet_emb")
    return Model(inputs=[anchor_inp, positive_inp, negative_inp],
                 outputs=merged, name="SiameseNet")


# ── Custom Triplet Loss ────────────────────────────────────────────────────────

def triplet_loss(margin: float = MARGIN):
    """
    Custom loss function.
    y_pred shape: (batch, EMBEDDING_DIM * 3)
    """
    def loss(y_true, y_pred):
        D = EMBEDDING_DIM
        emb_a = y_pred[:, :D]
        emb_p = y_pred[:, D:2*D]
        emb_n = y_pred[:, 2*D:]

        d_pos = K.sum(K.square(emb_a - emb_p), axis=1)   # (batch,)
        d_neg = K.sum(K.square(emb_a - emb_n), axis=1)

        loss_val = K.maximum(d_pos - d_neg + margin, 0.0)
        return K.mean(loss_val)
    return loss


# ── Data Pipeline ──────────────────────────────────────────────────────────────

def load_img(path: str) -> np.ndarray:
    img = keras_image.load_img(path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img)
    return preprocess_input(arr)


def build_class_index(split: str = "train") -> dict:
    """Returns {class_name: [list of image paths]}."""
    split_dir = SUBSET_DIR / split
    index = {}
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        paths = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg"))
        if len(paths) >= 2:
            index[cls_dir.name] = [str(p) for p in paths]
    return index


def generate_triplets(class_index: dict, n_per_class: int = TRIPLETS_PER_CLASS):
    """
    Mine semi-hard triplets:
        anchor  ← random image from class C
        positive← different random image from class C
        negative← random image from class ≠ C
    """
    classes = list(class_index.keys())
    triplets = []
    for cls in classes:
        neg_classes = [c for c in classes if c != cls]
        for _ in range(n_per_class):
            a, p = random.sample(class_index[cls], 2)
            neg_cls  = random.choice(neg_classes)
            n        = random.choice(class_index[neg_cls])
            triplets.append((a, p, n))
    random.shuffle(triplets)
    return triplets


def triplet_generator(class_index: dict, batch_size: int = BATCH_SIZE):
    """Infinite generator yielding batches of triplets."""
    while True:
        triplets = generate_triplets(class_index)
        for i in range(0, len(triplets) - batch_size, batch_size):
            batch = triplets[i: i + batch_size]
            anchors   = np.array([load_img(t[0]) for t in batch])
            positives = np.array([load_img(t[1]) for t in batch])
            negatives = np.array([load_img(t[2]) for t in batch])
            # Dummy y (loss ignores it; shape must match batch)
            y_dummy   = np.zeros((len(batch), EMBEDDING_DIM * 3))
            yield [anchors, positives, negatives], y_dummy


# ── Training ───────────────────────────────────────────────────────────────────

def train_siamese():
    print("[siamese] Building model …")
    emb_net     = build_embedding_network()
    siamese_net = build_siamese_model(emb_net)

    siamese_net.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=triplet_loss(MARGIN)
    )
    siamese_net.summary()

    print("[siamese] Building data index …")
    train_index = build_class_index("train")
    val_index   = build_class_index("val")

    n_train_triplets = sum(len(v) for v in train_index.values()) * TRIPLETS_PER_CLASS
    steps_per_epoch  = max(1, n_train_triplets // BATCH_SIZE)
    val_steps        = max(1, steps_per_epoch // 5)

    callbacks = [
        EarlyStopping(monitor="loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(
            str(MODELS_DIR / "siamese_best.h5"),
            save_best_only=True, monitor="loss", verbose=1
        ),
        ReduceLROnPlateau(monitor="loss", factor=0.5, patience=2, verbose=1)
    ]

    print(f"[siamese] Training for {EPOCHS} epochs, "
          f"{steps_per_epoch} steps/epoch …")
    history = siamese_net.fit(
        triplet_generator(train_index, BATCH_SIZE),
        steps_per_epoch=steps_per_epoch,
        validation_data=triplet_generator(val_index, BATCH_SIZE),
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save embedding network separately (used at inference time)
    emb_net.save(MODELS_DIR / "siamese_embedding_net.h5")
    print(f"[siamese] Embedding network saved → {MODELS_DIR / 'siamese_embedding_net.h5'}")

    return history


def extract_siamese_embeddings():
    """
    After training, regenerate the embedding index using the Siamese model.
    Saves results/embeddings_siamese_train.npz
    """
    from tensorflow.keras.models import load_model

    print("[siamese] Extracting Siamese embeddings for index …")
    emb_net = load_model(MODELS_DIR / "siamese_embedding_net.h5", compile=False)

    for split in ("train", "val"):
        class_index = build_class_index(split)
        all_paths, all_labels, all_embs = [], [], []
        class_names = sorted(class_index.keys())
        cls_to_idx  = {c: i for i, c in enumerate(class_names)}

        for cls, paths in tqdm(class_index.items(), desc=f"Embedding {split}"):
            imgs  = np.array([load_img(p) for p in paths])
            embs  = emb_net.predict(imgs, verbose=0, batch_size=BATCH_SIZE)
            all_embs.extend(embs)
            all_labels.extend([cls_to_idx[cls]] * len(paths))
            all_paths.extend(paths)

        emb_arr = np.array(all_embs)
        # L2 normalise
        emb_arr = emb_arr / (np.linalg.norm(emb_arr, axis=1, keepdims=True) + 1e-8)
        out = RESULTS_DIR / f"embeddings_siamese_{split}.npz"
        np.savez_compressed(
            out,
            embeddings=emb_arr,
            labels=np.array(all_labels),
            paths=np.array(all_paths),
            class_names=np.array(class_names)
        )
        print(f"[siamese] Saved → {out}  shape={emb_arr.shape}")


def main():
    history = train_siamese()
    extract_siamese_embeddings()
    print("\n✅ Siamese network training complete.")


if __name__ == "__main__":
    main()
