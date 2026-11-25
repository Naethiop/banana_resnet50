"""Stage 5: frozen-base ResNet50 baseline, and the training-only augmentation definition.

    python -m src.train                      # baseline: train the new head on cached features
    python -m src.train --show-augmentation  # save a picture of the augmentation on training images

Baseline = ResNet50 (ImageNet weights) frozen, plus a new softmax layer (Chapter Three, stage 1 of
training). Because the base is frozen, its output for each image does not change, so it is computed once
(src/features.py) and only the new layer is trained. This is mathematically the same as training the
head on top of the frozen base, and takes seconds instead of hours on a CPU. Augmentation is defined
here and applied to training images from Stage 6, when images are fed through the network end to end.

Only the training and validation sets are used here. The test set is never read for labels or metrics.
Two settings are compared on VALIDATION data only: no class weights, and balanced class weights.
"""
import argparse
import json
import platform
import time
from datetime import date

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from src import config


# ---------- pure helpers (tested without TensorFlow) ----------
def class_weight_dict(y, n_classes):
    w = compute_class_weight("balanced", classes=np.arange(n_classes), y=y)
    return {i: float(v) for i, v in enumerate(w)}


def evaluate_predictions(y_true, y_pred, names, groups=None):
    labels = list(range(len(names)))
    rep = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    for i, name in enumerate(names):
        r = rep[name]
        per_class[name] = {"precision": float(r["precision"]), "recall": float(r["recall"]),
                           "f1": float(r["f1-score"]), "images": int(r["support"]),
                           "photograph_groups": int(len(set(groups[np.asarray(y_true) == i])))
                           if groups is not None else None}
    return {"accuracy": float(rep["accuracy"]), "macro_f1": float(rep["macro avg"]["f1-score"]),
            "per_class": per_class, "confusion": cm.tolist()}


def choose_setting(results):
    """Highest validation macro-F1; ties go to 'none' (the simpler setting)."""
    order = sorted(results, key=lambda k: (-results[k]["macro_f1"], k != "none"))
    return order[0]


# ---------- Keras parts ----------
def build_augmentation():
    """Training-only augmentation (flips, small rotations, brightness). Active only with training=True."""
    import keras
    a = config.AUGMENTATION
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(a["rotation_factor"], fill_mode="reflect"),
        keras.layers.RandomBrightness(a["brightness_factor"], value_range=(0, 255)),
    ], name="augmentation")


def build_head(n_classes):
    import keras
    return keras.Sequential([
        keras.Input(shape=(2048,)),
        keras.layers.Dropout(config.HEAD_DROPOUT),
        keras.layers.Dense(n_classes, activation="softmax"),
    ], name="head")


def train_head(x_tr, y_tr, x_va, y_va, n_classes, class_weight):
    import keras
    keras.utils.set_random_seed(config.SEED)
    head = build_head(n_classes)
    head.compile(optimizer=keras.optimizers.Adam(config.HEAD_LEARNING_RATE),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE,
                                         restore_best_weights=True)
    hist = head.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=config.MAX_EPOCHS_HEAD,
                    batch_size=config.HEAD_BATCH_SIZE, class_weight=class_weight,
                    callbacks=[stop], verbose=0)
    return head, hist.history


def assemble_full_model(head):
    """Frozen ResNet50 base + trained head, taking RGB images that went through preprocess_batch."""
    import keras
    base = keras.applications.ResNet50(include_top=False, weights="imagenet", pooling="avg",
                                       input_shape=tuple(config.IMAGE_SIZE) + (3,))
    base.trainable = False
    inputs = keras.Input(shape=tuple(config.IMAGE_SIZE) + (3,))
    outputs = head(base(inputs, training=False))
    return keras.Model(inputs, outputs, name="resnet50_banana_baseline")


def show_augmentation():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.features import load_split_table
    from src.preprocessing import load_rgb

    import keras
    keras.utils.set_random_seed(config.SEED)
    aug = build_augmentation()
    df = load_split_table()
    tr = df[df.assigned_split == "train"]
    pick = tr.groupby("label").head(1).head(5)
    fig, axes = plt.subplots(len(pick), 4, figsize=(8, 2 * len(pick)))
    for r, (_, row) in enumerate(pick.iterrows()):
        img = load_rgb(row.path)
        axes[r, 0].imshow(img.astype("uint8"))
        axes[r, 0].set_title("original", fontsize=7)
        for c in range(1, 4):
            out = np.asarray(aug(img[None], training=True))[0]
            axes[r, c].imshow(np.clip(out, 0, 255).astype("uint8"))
            axes[r, c].set_title("augmented", fontsize=7)
        for c in range(4):
            axes[r, c].axis("off")
    config.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    out = config.EVIDENCE_DIR / "08_augmentation_examples.png"
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")


def run_baseline():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import keras
    from src.preprocessing import load_rgb, preprocess_batch
    from src.features import load_split_table

    if not config.FEATURES_FILE.exists():
        raise SystemExit("Run python -m src.features first.")
    names = json.loads(config.CLASS_NAMES_SPLIT_FILE.read_text(encoding="utf-8"))
    n = len(names)
    from src.features import load_features
    d = load_features()
    feats, files, labels, groups, splits = (d[k] for k in ("features", "files", "labels", "groups", "splits"))
    y = np.array([names.index(lab) for lab in labels])
    tr, va = splits == "train", splits == "val"      # the test rows are deliberately never used here
    x_tr, y_tr, x_va, y_va, g_va = feats[tr], y[tr], feats[va], y[va], groups[va]

    results, heads, histories = {}, {}, {}
    for setting in ("none", "balanced"):
        cw = class_weight_dict(y_tr, n) if setting == "balanced" else None
        head, hist = train_head(x_tr, y_tr, x_va, y_va, n, cw)
        pred = np.argmax(head.predict(x_va, verbose=0), axis=1)
        results[setting] = evaluate_predictions(y_va, pred, names, g_va)
        results[setting]["epochs_run"] = len(hist["loss"])
        results[setting]["class_weights"] = cw
        heads[setting], histories[setting] = head, hist
        print(f"{setting:9s} val accuracy {results[setting]['accuracy']:.3f}  "
              f"macro-F1 {results[setting]['macro_f1']:.3f}  epochs {results[setting]['epochs_run']}")
    best = choose_setting(results)
    print(f"Chosen on validation macro-F1: {best}")

    # curves and confusion matrix (validation)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    h = histories[best]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="validation"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(h["accuracy"], label="train"); ax[1].plot(h["val_accuracy"], label="validation"); ax[1].set_title("Accuracy"); ax[1].legend()
    fig.tight_layout(); fig.savefig(config.REPORTS_DIR / "baseline_curves.png", dpi=110); plt.close(fig)
    cm = np.array(results[best]["confusion"])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7); ax.set_yticklabels(names, fontsize=7)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual (validation)")
    fig.tight_layout(); fig.savefig(config.REPORTS_DIR / "baseline_confusion_val.png", dpi=110); plt.close(fig)

    # assemble the full model and confirm it agrees with the head on cached features
    full = assemble_full_model(heads[best])
    df = load_split_table()
    val_df = df[df.assigned_split == "val"].head(8)
    batch = preprocess_batch(np.stack([load_rgb(p) for p in val_df.path]))
    p_full = np.asarray(full.predict(batch, verbose=0))
    idx = [int(np.where(files == f)[0][0]) for f in val_df.file]
    p_head = np.asarray(heads[best].predict(feats[idx], verbose=0))
    diff = float(np.abs(p_full - p_head).max())
    print(f"Full model versus head on cached features, max difference: {diff:.6f}")
    if diff > 5e-3:
        raise SystemExit("Full model and cached-feature head disagree; do not use this baseline.")
    config.MODELS_DIR.mkdir(exist_ok=True)
    full.save(str(config.BASELINE_MODEL_FILE))

    meta = {"stage": "5 baseline (frozen ResNet50 base + new head)", "date": str(date.today()),
            "python": platform.python_version(), "keras": keras.__version__,
            "seed": config.SEED, "class_names": names, "class_weight_setting": best,
            "head_learning_rate": config.HEAD_LEARNING_RATE, "dropout": config.HEAD_DROPOUT,
            "epochs_run": results[best]["epochs_run"],
            "preprocessing": "src/preprocessing.py: centre-crop, 224x224, ResNet50 caffe-mode (BGR, mean subtraction)",
            "train_images": int(tr.sum()), "val_images": int(va.sum()),
            "validation_accuracy": results[best]["accuracy"], "validation_macro_f1": results[best]["macro_f1"],
            "note": "validation results only; the test set has not been used"}
    try:
        import tensorflow as tf
        meta["tensorflow"] = tf.__version__
    except Exception:  # noqa: BLE001
        pass
    config.BASELINE_META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # report
    L = ["# Baseline Validation Report (Stage 5)\n",
         "Frozen ResNet50 (ImageNet) base with a new softmax layer. **Validation data only. The test set has not been used.**\n",
         "## Settings compared on validation data\n",
         "| Class weights | Epochs run | Validation accuracy | Validation macro-F1 |", "|---|---|---|---|"]
    for k, r in results.items():
        L.append(f"| {k}{' (chosen)' if k == best else ''} | {r['epochs_run']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} |")
    L.append(f"\nThe validation set has {int(va.sum())} images from {len(set(g_va))} photograph groups, "
             "so every figure below is a rough indication, not a precise estimate.\n")
    L.append(f"## Per-class validation results ({best})\n")
    L.append("| Class | Precision | Recall | F1 | Validation images | Photograph groups |\n|---|---|---|---|---|---|")
    for name, r in results[best]["per_class"].items():
        L.append(f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['images']} | {r['photograph_groups']} |")
    L.append("\n## Confusion matrix (rows actual, columns predicted)\n")
    L.append("| | " + " | ".join(names) + " |\n|---|" + "---|" * n)
    for i, name in enumerate(names):
        L.append(f"| **{name}** | " + " | ".join(str(v) for v in cm[i]) + " |")
    L.append("\nFigures: `reports/baseline_curves.png`, `reports/baseline_confusion_val.png`.")
    L.append("\n## Notes\n- Training accuracy is not evidence of generalisation; use the validation columns.\n"
             "- Class weights were compared on validation macro-F1 only.\n"
             "- Augmentation is not used in this baseline (features are cached); it is introduced in Stage 6.")
    config.BASELINE_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {config.BASELINE_REPORT}, baseline_curves.png, baseline_confusion_val.png, "
          f"{config.BASELINE_MODEL_FILE.name}, {config.BASELINE_META_FILE.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-augmentation", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if args.show_augmentation:
        show_augmentation()
    else:
        run_baseline()
    print(f"Finished in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
