"""
transfer_learning.py
--------------------
Fine-tunes the last N layers of ResNet50 on the fashion subset
as a classification task. The fine-tuned backbone is then used
as an improved feature extractor for similarity search.

Usage:
    python src/transfer_learning.py
"""

import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Config ─────────────────────────────────────────────────────────────────────
SUBSET_DIR    = Path("data/subset")
MODELS_DIR    = Path("models")
RESULTS_DIR   = Path("results")
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS_FROZEN = 5     # train only new head with base frozen
EPOCHS_FINE   = 10    # then unfreeze last UNFREEZE_LAYERS layers
UNFREEZE_LAYERS = 30  # ResNet50 has ~175 layers; unfreeze last 30
LEARNING_RATE   = 1e-4
FINE_LR         = 1e-5
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def build_classifier(num_classes: int) -> tuple[Model, Model]:
    """
    Returns:
        full_model      – classifier (for training)
        feature_model   – embedding extractor (for retrieval)
    """
    base = ResNet50(weights="imagenet", include_top=False,
                    input_shape=(*IMG_SIZE, 3), pooling="avg")
    base.trainable = False   # phase 1: freeze everything

    x = base.output
    x = layers.Dense(512, activation="relu", name="fc_512")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    full_model    = Model(inputs=base.input, outputs=out, name="ResNet50_FT")
    feature_model = Model(inputs=base.input, outputs=base.output, name="ResNet50_Emb")

    return full_model, feature_model


def get_data_generators() -> tuple:
    """ImageDataGenerator with augmentation for train, normalisation for val."""
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15,
    )
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_flow = train_gen.flow_from_directory(
        SUBSET_DIR / "train", target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True
    )
    val_flow = val_gen.flow_from_directory(
        SUBSET_DIR / "val", target_size=IMG_SIZE,
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
    )
    return train_flow, val_flow


def phase1_train(model: Model, train_flow, val_flow) -> None:
    """Train only the new classification head with base frozen."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print("\n[transfer_learning] Phase 1 — training head only")
    model.fit(
        train_flow, validation_data=val_flow,
        epochs=EPOCHS_FROZEN,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
    )


def phase2_finetune(model: Model, train_flow, val_flow) -> None:
    """Unfreeze last UNFREEZE_LAYERS layers and fine-tune."""
    # Unfreeze last N layers of base (index -UNFREEZE_LAYERS onwards)
    base = model.get_layer("resnet50")
    for layer in base.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True
    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"[transfer_learning] Phase 2 — fine-tuning {trainable} trainable layers")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(FINE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    ckpt = ModelCheckpoint(
        MODELS_DIR / "resnet50_finetuned.h5",
        save_best_only=True, monitor="val_accuracy", verbose=1
    )
    lr_sched = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    model.fit(
        train_flow, validation_data=val_flow,
        epochs=EPOCHS_FINE,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True), ckpt, lr_sched]
    )


def save_feature_model(full_model: Model):
    """Strip classification head, save embedding extractor."""
    # Rebuild feature model from fine-tuned base
    base = full_model.get_layer("resnet50")
    feat_model = Model(inputs=base.input, outputs=base.output,
                       name="ResNet50_FineTuned_Emb")
    feat_model.save(MODELS_DIR / "feature_model_finetuned.h5")
    print(f"[transfer_learning] Saved → {MODELS_DIR / 'feature_model_finetuned.h5'}")


def main():
    train_flow, val_flow = get_data_generators()
    num_classes = train_flow.num_classes
    print(f"[transfer_learning] Classes: {num_classes}  |  "
          f"Train: {train_flow.n}  |  Val: {val_flow.n}")

    full_model, _ = build_classifier(num_classes)

    phase1_train(full_model, train_flow, val_flow)
    phase2_finetune(full_model, train_flow, val_flow)
    save_feature_model(full_model)

    print("\n✅ Transfer learning complete. Model saved to models/")


if __name__ == "__main__":
    main()
