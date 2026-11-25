"""Stage 6b: two-stage fine-tuning of ResNet50, compared with the head-only model in group cross-validation.

    python -m src.finetune --time-check   # measures speed on this laptop, trains nothing useful
    python -m src.finetune                # one repeat of the 4-fold group cross-validation (runs for an hour or more)
    python -m src.finetune --resume       # continue after an interruption

Stage 1 (per fold): train the new softmax layer on cached features of the training images plus their
augmented copies, with balanced class weights (variant D of Stage 6a).
Stage 2 (per fold): unfreeze the last ResNet50 stage (layers named conv5_*, batch-norm layers stay frozen)
and train it together with the head at a low learning rate on freshly augmented training images,
for a fixed number of epochs. The test fold is only used for the final predictions of the fold.
Both models are scored on the SAME folds (repeat 0 of Stage 6a), so the comparison is paired.
Only train and validation images are used; the test set is never touched.
Keep the laptop plugged in and prevent it from sleeping while this runs.
"""
import argparse
import json
import time

import numpy as np

from src import config
from src.cv import ci, fit_head, group_bootstrap, group_correctness, load_pool, make_folds, metrics
from src.preprocessing import load_rgb, preprocess_batch


def load_images(paths):
    """All pool images as uint8 arrays (N, 224, 224, 3); about 120 MB for 783 images."""
    return np.stack([load_rgb(p).astype(np.uint8) for p in paths])


def build_finetune_model(head):
    """Frozen ResNet50 except the last stage, plus the given head. Takes preprocess_batch output."""
    import keras
    base = keras.applications.ResNet50(include_top=False, weights="imagenet", pooling="avg",
                                       input_shape=tuple(config.IMAGE_SIZE) + (3,))
    for layer in base.layers:
        layer.trainable = (layer.name.startswith(config.FT_UNFREEZE_PREFIX)
                           and not isinstance(layer, keras.layers.BatchNormalization))
    inputs = keras.Input(shape=tuple(config.IMAGE_SIZE) + (3,))
    model = keras.Model(inputs, head(base(inputs)))
    model.compile(optimizer=keras.optimizers.Adam(config.FINE_TUNE_LEARNING_RATE),
                  loss="sparse_categorical_crossentropy")
    return model, base


def train_epochs(model, images, y, train_idx, class_weight, epochs, seed, label=""):
    from src.train import build_augmentation
    aug = build_augmentation()
    rng = np.random.default_rng(seed)
    bs = config.BATCH_SIZE
    for ep in range(epochs):
        t0, losses = time.time(), []
        order = rng.permutation(train_idx)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            xb = np.asarray(aug(images[idx].astype("float32"), training=True))
            xb = preprocess_batch(np.clip(xb, 0, 255))
            loss = model.train_on_batch(xb, y[idx], class_weight=class_weight)
            losses.append(float(np.ravel(loss)[0]))
        print(f"  {label} epoch {ep + 1}/{epochs}: training loss {np.mean(losses):.3f}, "
              f"{time.time() - t0:.0f} s", flush=True)


def predict_images(model, images, idx):
    out = []
    for s in range(0, len(idx), config.BATCH_SIZE):
        out.append(np.asarray(model.predict(preprocess_batch(images[idx[s:s + config.BATCH_SIZE]].astype("float32")),
                                            verbose=0)))
    return np.concatenate(out)


def time_check(P):
    import keras
    from src.train import build_augmentation, build_head
    paths = list(P["table"].path[:64])
    images = load_images(paths)
    y = P["y"][:64]
    head = build_head(len(P["names"]))
    model, _ = build_finetune_model(head)
    n_train = sum(1 for _ in model.trainable_weights)
    aug = build_augmentation()
    bs = config.BATCH_SIZE
    def step(i):
        xb = np.asarray(aug(images[i:i + bs].astype("float32"), training=True))
        model.train_on_batch(preprocess_batch(np.clip(xb, 0, 255)), y[i:i + bs])
    step(0)                                           # warm-up (graph tracing)
    t0 = time.time()
    for i in range(bs, 64, bs):
        step(i)
    per_img = (time.time() - t0) / (64 - bs)
    t1 = time.time()
    predict_images(model, images, np.arange(32))
    inf = (time.time() - t1) / 32
    n_tr = int(0.75 * len(P["y"]))
    print(f"Trainable weight tensors in the fine-tuning model: {n_train}")
    print(f"Fine-tuning speed: {per_img:.2f} s per image. Inference: {inf:.2f} s per image.")
    print(f"Projected: one epoch over about {n_tr} images = {n_tr * per_img / 60:.1f} min; "
          f"one fold ({config.FT_EPOCHS} epochs) = {config.FT_EPOCHS * n_tr * per_img / 60:.0f} min; "
          f"{config.CV_FOLDS} folds = {config.CV_FOLDS * config.FT_EPOCHS * n_tr * per_img / 60:.0f} min.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-check", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not config.FEATURES_FILE.exists() or not config.AUG_FEATURES_FILE.exists():
        raise SystemExit("Run python -m src.features and python -m src.features --augment 4 first.")
    import keras
    P = load_pool()
    names, X, y, groups, aug = P["names"], P["X"], P["y"], P["groups"], P["aug"]
    n = len(names)
    if args.time_check:
        time_check(P)
        return
    print("Loading pool images...")
    images = load_images(list(P["table"].path))
    folds = make_folds(y, groups, config.CV_FOLDS, config.SEED)      # repeat 0 of Stage 6a
    proba_head = np.zeros((len(y), n))
    proba_ft = np.zeros((len(y), n))
    done, t_start = [], time.time()
    if args.resume and config.FT_PARTIAL.exists():
        z = np.load(config.FT_PARTIAL)
        proba_head, proba_ft, done = z["head"], z["ft"], list(z["done"])
        print(f"Resuming; folds already done: {done}")
    for k, (tr, te) in enumerate(folds):
        if k in done:
            continue
        print(f"Fold {k + 1}/{len(folds)}: {len(tr)} training images, {len(te)} test images", flush=True)
        head, cw = fit_head(X, y, tr, n, True, aug)                     # stage 1 (variant D)
        proba_head[te] = head.predict(X[te], verbose=0)
        model, _ = build_finetune_model(head)                            # stage 2
        train_epochs(model, images, y, tr, cw, config.FT_EPOCHS, config.SEED + k, label=f"fold {k + 1}")
        proba_ft[te] = predict_images(model, images, te)
        done.append(k)
        np.savez(config.FT_PARTIAL, head=proba_head, ft=proba_ft, done=np.array(done))
        print(f"  fold {k + 1} finished; elapsed {(time.time() - t_start) / 60:.0f} min", flush=True)
        keras.backend.clear_session()

    pred_h, pred_f = proba_head.argmax(axis=1), proba_ft.argmax(axis=1)
    preds = {"head only (stage 1)": pred_h, "fine-tuned (stage 2)": pred_f}
    boot = group_bootstrap(y, groups, preds, n, config.BOOTSTRAP_SAMPLES, config.SEED)
    mh, mf = metrics(y, pred_h, n), metrics(y, pred_f, n)
    dl, dh = ci(boot["fine-tuned (stage 2)"]["macro_f1"] - boot["head only (stage 1)"]["macro_f1"])
    justified = dl > 0
    L = ["# Fine-Tuning Report (Stage 6b)\n",
         "Two-stage training compared with the head-only model on the SAME group cross-validation folds "
         "(one repeat, 4 folds). **Train and validation images only; the test set was not used.**\n",
         "## Setup\n",
         f"- Stage 1: head on cached features plus {config.AUG_COPIES} augmented copies, balanced class weights (variant D)",
         f"- Stage 2: layers named `{config.FT_UNFREEZE_PREFIX}*` unfrozen (batch-norm layers frozen), Adam learning rate "
         f"{config.FINE_TUNE_LEARNING_RATE}, {config.FT_EPOCHS} fixed epochs, fresh augmentation every epoch, batch size {config.BATCH_SIZE}",
         f"- Seed {config.SEED}; one repeat of the cross-validation, so fold assignment variability is not averaged out",
         f"- Total run time {(time.time() - t_start) / 60:.0f} minutes",
         f"- Intervals: 95 percent, {config.BOOTSTRAP_SAMPLES} resamples of whole photograph groups within each class, "
         "same resamples for both models\n",
         "## Comparison\n",
         "| Model | Accuracy | Macro-F1 | Macro-F1 95% interval |", "|---|---|---|---|"]
    for nm, m in (("head only (stage 1)", mh), ("fine-tuned (stage 2)", mf)):
        lo, hi = ci(boot[nm]["macro_f1"])
        L.append(f"| {nm} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {lo:.3f} to {hi:.3f} |")
    L.append(f"\nPaired macro-F1 difference (fine-tuned minus head only): "
             f"{boot['fine-tuned (stage 2)']['macro_f1'].mean() - boot['head only (stage 1)']['macro_f1'].mean():+.3f} "
             f"(95% interval {dl:+.3f} to {dh:+.3f}).\n")
    L.append("Selection rule (as in Stage 6a): fine-tuning is justified only if this interval lies entirely above zero. "
             f"**Result: {'justified' if justified else 'not justified; keep the head-only model'}.**\n")
    for nm, m, pr in (("head only (stage 1)", mh, pred_h), ("fine-tuned (stage 2)", mf, pred_f)):
        gc = group_correctness(y, groups, pr, n)
        L += [f"## Per-class results: {nm}\n",
              "| Class | Recall | Recall 95% interval | Precision | Photograph groups | Groups mostly correct |",
              "|---|---|---|---|---|---|"]
        for c, name in enumerate(names):
            lo, hi = ci(boot[nm]["recall"][:, c])
            L.append(f"| {name} | {m['recall'][c]:.3f} | {lo:.3f} to {hi:.3f} | {m['precision'][c]:.3f} | {gc[c][0]} | {gc[c][1]} |")
        L += ["", "Confusion matrix (rows actual, columns predicted):\n",
              "| | " + " | ".join(names) + " |", "|---|" + "---|" * n]
        for i, name in enumerate(names):
            L.append(f"| **{name}** | " + " | ".join(str(v) for v in m["confusion"][i]) + " |")
        L.append("")
    L += ["## Notes\n",
          "- The number of epochs was fixed in advance; it was not tuned on the test folds.",
          "- Cross-validation scores describe this dataset only and rest on very few independent photographs.",
          "- One repeat only: a different fold assignment could change the result."]
    config.FT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {config.FT_REPORT}")
    print(f"Head only macro-F1 {mh['macro_f1']:.3f}; fine-tuned {mf['macro_f1']:.3f}; "
          f"paired difference interval {dl:+.3f} to {dh:+.3f}; fine-tuning {'justified' if justified else 'not justified'}")


if __name__ == "__main__":
    main()
