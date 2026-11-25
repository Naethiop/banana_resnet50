"""Stage 7: train the final model, freeze it, then evaluate it ONCE on the test set.

    python -m src.evaluate                  # final model trained on the train+val pool (recommended)
    python -m src.evaluate --train-on train # final model trained on the train split only

Final model = the variant chosen in Stage 6: frozen ResNet50 (ImageNet) + new softmax layer, trained on
cached features of the training images plus augmented copies, with balanced class weights, for the fixed
number of epochs used in cross-validation. Fine-tuning was not justified in Stage 6b.

Order of operations: (1) train the head, (2) assemble and SAVE the model package, (3) only then read the test
images and labels. A lock file records that the test set has been evaluated; the script refuses to evaluate
again unless --allow-rerun is given, because repeated looks at the test set would invalidate it.
"""
import argparse
import json
import time
from datetime import datetime

import numpy as np

from src import config
from src.cv import ci, fit_head, group_bootstrap, group_correctness, load_pool, metrics
from src.model_io import save_package, sha256_of
from src.preprocessing import load_rgb, preprocess_batch


def already_evaluated(lock_path=config.TEST_LOCK_FILE):
    return lock_path.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-on", choices=("pool", "train"), default="pool")
    ap.add_argument("--allow-rerun", action="store_true")
    args = ap.parse_args()
    if already_evaluated() and not args.allow_rerun:
        raise SystemExit(f"The test set was already evaluated ({config.TEST_LOCK_FILE}). "
                         "Do not evaluate it again. Use --allow-rerun only if you must, and report that you did.")
    if not config.AUG_FEATURES_FILE.exists():
        raise SystemExit("Run python -m src.features --augment 4 first.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.features import load_features
    from src.train import assemble_full_model

    t0 = time.time()
    P = load_pool()                                   # train+val only
    names, X, y, groups, aug = P["names"], P["X"], P["y"], P["groups"], P["aug"]
    n = len(names)
    table = P["table"]
    if args.train_on == "train":
        train_idx = np.where(table.assigned_split.to_numpy() == "train")[0]
    else:
        train_idx = np.arange(len(y))
    n_groups = len(np.unique(groups[train_idx]))
    print(f"Training the final head on {len(train_idx)} images in {n_groups} photograph groups ({args.train_on})")
    head, cw = fit_head(X, y, train_idx, n, True, aug)
    model = assemble_full_model(head)

    cv_info = {}
    if config.CV_JSON.exists():
        cv_info = json.loads(config.CV_JSON.read_text(encoding="utf-8"))
    meta = {
        "stage": "7 final model (frozen ResNet50 + new head)",
        "trained_on": f"{args.train_on}: {len(train_idx)} images, {n_groups} photograph groups",
        "augmented_copies_per_image": config.AUG_COPIES, "class_weights": "balanced",
        "class_weight_values": cw, "head_epochs": config.CV_EPOCHS, "head_learning_rate": config.HEAD_LEARNING_RATE,
        "dropout": config.HEAD_DROPOUT, "seed": config.SEED, "fine_tuning": "not used (Stage 6b: not justified)",
        "preprocessing": "src/preprocessing.py: EXIF orientation, RGB, centre-crop square, 224x224, ResNet50 caffe-mode "
                         "(BGR, ImageNet mean subtraction); the model expects the output of preprocess_batch",
        "low_confidence_threshold": None,
        "cross_validation": cv_info.get("variants", {}).get("D augmented + weights"),
        "note": "model frozen before the test set was read"}
    save_package(model, names, meta)
    print(f"Model package saved and frozen ({time.time() - t0:.0f} s). Now reading the test set.")

    # ---- test set: read only now ----
    d = load_features()
    tab = d["table"]
    test_mask = tab.assigned_split.to_numpy() == "test"
    test = tab[test_mask].reset_index(drop=True)
    y_t = np.array([names.index(lab) for lab in test.label])
    g_t = test.group_id.to_numpy()
    images = np.stack([load_rgb(p) for p in test.path])
    proba = np.concatenate([np.asarray(model.predict(preprocess_batch(images[i:i + 16]), verbose=0))
                            for i in range(0, len(images), 16)])
    pred = proba.argmax(axis=1)
    diff = float(np.abs(proba - head.predict(d["features"][test_mask], verbose=0)).max())
    print(f"Full model versus cached-feature head on the test images, max difference {diff:.6f}")
    if diff > 5e-3:
        raise SystemExit("Evaluation path and cached-feature path disagree; investigate before reporting.")

    m = metrics(y_t, pred, n)
    boot = group_bootstrap(y_t, g_t, {"final": pred}, n, config.BOOTSTRAP_SAMPLES, config.SEED)
    gc = group_correctness(y_t, g_t, pred, n)
    conf = proba.max(axis=1)
    wrong = test.assign(true=[names[i] for i in y_t], predicted=[names[i] for i in pred], confidence=conf.round(3))
    wrong[y_t != pred][["file", "true", "predicted", "confidence", "group_id"]].to_csv(
        config.TEST_MISCLASSIFIED_CSV, index=False)

    cm = m["confusion"]
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7); ax.set_yticklabels(names, fontsize=7)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual (test)")
    fig.tight_layout(); fig.savefig(config.TEST_CONFUSION_PNG, dpi=110); plt.close(fig)

    lo, hi = ci(boot["final"]["macro_f1"])
    L = ["# Final Test Evaluation (Stage 7)\n",
         "The model was trained and saved before the test set was read, and the test set was evaluated once.\n",
         "## Model\n",
         "- Frozen ResNet50 (ImageNet weights) plus a new softmax layer; augmentation and balanced class weights; "
         f"{config.CV_EPOCHS} epochs. Fine-tuning was not used (Stage 6b: not justified).",
         f"- Trained on: {meta['trained_on']}",
         f"- Test set: {len(test)} images in {len(np.unique(g_t))} photograph groups, never used for training, "
         "model selection or threshold choice\n",
         "## Test results\n",
         f"- Accuracy: **{m['accuracy']:.3f}**",
         f"- Macro-F1: **{m['macro_f1']:.3f}** (95% group-bootstrap interval {lo:.3f} to {hi:.3f})",
         "- Cross-validation estimate for the same method on the train and validation pool (Stage 6a, variant D): "
         + (f"macro-F1 {cv_info['variants']['D augmented + weights']['macro_f1']:.3f}, accuracy "
            f"{cv_info['variants']['D augmented + weights']['accuracy']:.3f}" if cv_info else "see reports/cv_report.md"),
         "\n| Class | Precision | Recall | Recall 95% interval | F1 | Test images | Photograph groups | Groups mostly correct |",
         "|---|---|---|---|---|---|---|---|"]
    for c, nm in enumerate(names):
        rlo, rhi = ci(boot["final"]["recall"][:, c])
        L.append(f"| {nm} | {m['precision'][c]:.3f} | {m['recall'][c]:.3f} | {rlo:.3f} to {rhi:.3f} | {m['f1'][c]:.3f} | "
                 f"{int((y_t == c).sum())} | {gc[c][0]} | {gc[c][1]} |")
    L += ["\n## Confusion matrix (rows actual, columns predicted)\n",
          "| | " + " | ".join(names) + " |", "|---|" + "---|" * n]
    for i, nm in enumerate(names):
        L.append(f"| **{nm}** | " + " | ".join(str(v) for v in cm[i]) + " |")
    L += [f"\nFigure: `reports/test_confusion_matrix.png`. Misclassified images: `reports/test_misclassified.csv` "
          f"({int((y_t != pred).sum())} of {len(test)}).\n",
          "## How to read this\n",
          f"- The test set holds only {len(np.unique(g_t))} distinct photographs; a class with 1 or 2 groups has a result that rests on 1 or 2 photographs.",
          "- Intervals are wide for the same reason, and a class with a single group has a meaningless interval.",
          "- The cross-validation estimate, which uses all 100 train and validation photograph groups, is the better guide to expected performance on this dataset.",
          "- Results describe this dataset only; other cameras, backgrounds and regions are untested.",
          "- Confidence scores are not calibrated probabilities."]
    config.TEST_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    config.TEST_LOCK_FILE.write_text(json.dumps({
        "evaluated_at": datetime.now().isoformat(timespec="seconds"), "train_on": args.train_on,
        "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "model_sha256": sha256_of(config.MODEL_FILE),
        "note": "The test set has been evaluated. Do not evaluate it again."}, indent=2), encoding="utf-8")
    print(f"Test accuracy {m['accuracy']:.3f}, macro-F1 {m['macro_f1']:.3f}. Wrote {config.TEST_REPORT}. "
          f"Locked. Total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
