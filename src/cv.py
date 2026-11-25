"""Stage 6a: repeated group-stratified cross-validation of the frozen-base head, with uncertainty.

    python -m src.features --augment 4   # once: cache features of augmented training copies
    python -m src.cv

Only the train and validation images are used (the test set is excluded in code). All images of a
photograph group are always on the same side of every fold, so a fold's test images are never copies
of its training images. Variants compared (all use the same cached ResNet50 features):
    A  head only                       (reference)
    B  head + balanced class weights
    C  head + augmented copies         (needs the augmented feature cache)
    D  head + augmented copies + balanced class weights
Selection rule, fixed in advance: A is kept unless another variant improves macro-F1 and the 95 percent
group-bootstrap interval of the paired difference from A lies entirely above zero.
"""
import argparse
import json
import time

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src import config


# ---------- pure helpers (tested without TensorFlow) ----------
def make_folds(y, groups, n_splits, seed):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups))


def run_cv(n_items, y, groups, predict_fn, n_classes, n_splits, repeats, seed):
    """Out-of-fold probabilities, shape (repeats, n_items, n_classes). predict_fn(train_idx, test_idx)."""
    out = np.zeros((repeats, n_items, n_classes))
    for r in range(repeats):
        for tr, te in make_folds(y, groups, n_splits, seed + r):
            out[r, te] = predict_fn(tr, te)
    return out


def metrics(y, pred, n):
    """Accuracy, macro-F1, per-class precision/recall/F1 from a confusion matrix."""
    cm = np.bincount(y * n + pred, minlength=n * n).reshape(n, n).astype(float)
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(axis=0), out=np.zeros(n), where=cm.sum(axis=0) > 0)
    rec = np.divide(tp, cm.sum(axis=1), out=np.zeros(n), where=cm.sum(axis=1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(n), where=(prec + rec) > 0)
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1.mean()),
            "precision": prec, "recall": rec, "f1": f1, "confusion": cm.astype(int)}


def group_bootstrap(y, groups, preds, n_classes, n_boot, seed):
    """Resample photograph groups with replacement within each class (same resamples for every variant).

    preds: {variant: predicted labels}. Returns {variant: {'macro_f1': (B,), 'recall': (B, n_classes)}}.
    """
    rng = np.random.default_rng(seed)
    members = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    by_class = {c: np.unique(groups[y == c]) for c in range(n_classes)}
    out = {k: {"macro_f1": np.zeros(n_boot), "recall": np.zeros((n_boot, n_classes))} for k in preds}
    for b in range(n_boot):
        idx = np.concatenate([members[g] for c in range(n_classes)
                              for g in rng.choice(by_class[c], size=len(by_class[c]), replace=True)])
        for k, p in preds.items():
            m = metrics(y[idx], p[idx], n_classes)
            out[k]["macro_f1"][b] = m["macro_f1"]
            out[k]["recall"][b] = m["recall"]
    return out


def ci(values, level=95):
    lo, hi = np.percentile(values, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(lo), float(hi)


def prefer_variant(boot, reference, candidate):
    """True if the candidate beats the reference with a 95 percent paired-difference interval above zero."""
    diff = boot[candidate]["macro_f1"] - boot[reference]["macro_f1"]
    return ci(diff)[0] > 0


def group_correctness(y, groups, pred, n_classes):
    """Per class: (number of groups, groups with more than half of their images classified correctly)."""
    out = {}
    for c in range(n_classes):
        gs = np.unique(groups[y == c])
        ok = sum(1 for g in gs if (pred[groups == g] == c).mean() > 0.5)
        out[c] = (len(gs), ok)
    return out


# ---------- Keras head used inside the folds ----------
def fit_head(X, y, train_idx, n_classes, weighted, aug=None):
    """Train the new softmax layer on cached features of the training images (plus augmented copies)."""
    import keras
    from src.train import build_head, class_weight_dict
    keras.backend.clear_session()
    keras.utils.set_random_seed(config.SEED)
    xtr, ytr = X[train_idx], y[train_idx]
    if aug is not None:
        af, ay, asrc = aug
        keep = np.isin(asrc, train_idx)
        xtr, ytr = np.concatenate([xtr, af[keep]]), np.concatenate([ytr, ay[keep]])
    cw = class_weight_dict(ytr, n_classes) if weighted else None
    head = build_head(n_classes)
    head.compile(optimizer=keras.optimizers.Adam(config.HEAD_LEARNING_RATE),
                 loss="sparse_categorical_crossentropy")
    head.fit(xtr, ytr, epochs=config.CV_EPOCHS, batch_size=config.HEAD_BATCH_SIZE,
             class_weight=cw, verbose=0)
    return head, cw


def keras_predict_fn(X, y, n_classes, weighted, aug=None):
    """aug = (aug_features, aug_labels, aug_source_positions) or None."""
    def fn(train_idx, test_idx):
        head, _ = fit_head(X, y, train_idx, n_classes, weighted, aug)
        return head.predict(X[test_idx], verbose=0)
    return fn


def load_pool():
    """Train+val pool (never the test set): features, labels, groups, file names, optional augmented features."""
    from src.features import load_features

    names = json.loads(config.CLASS_NAMES_SPLIT_FILE.read_text(encoding="utf-8"))
    d = load_features()
    pool = d["splits"] != "test"                          # the test set is excluded here, in code
    X, files, groups = d["features"][pool], d["files"][pool], d["groups"][pool]
    y = np.array([names.index(lab) for lab in d["labels"][pool]])
    assert "test" not in set(d["splits"][pool])
    pos = {f: i for i, f in enumerate(files)}
    aug = None
    if config.AUG_FEATURES_FILE.exists():
        z = np.load(config.AUG_FEATURES_FILE, allow_pickle=False)
        keep = [i for i, f in enumerate(z["src_files"]) if str(f) in pos]
        asrc = np.array([pos[str(z["src_files"][i])] for i in keep])
        aug = (z["features"][keep], y[asrc], asrc)
    return {"names": names, "X": X, "y": y, "groups": groups, "files": files, "aug": aug,
            "table": d["table"][pool].reset_index(drop=True)}


def main():
    import matplotlib
    matplotlib.use("Agg")
    argparse.ArgumentParser().parse_args()
    if not config.FEATURES_FILE.exists():
        raise SystemExit("Run python -m src.features first.")
    P = load_pool()
    names, X, y, groups, aug = P["names"], P["X"], P["y"], P["groups"], P["aug"]
    files = P["files"]
    n = len(names)
    if aug is not None:
        print(f"Using {len(aug[0])} augmented feature rows")
    else:
        print("No augmented feature cache (run python -m src.features --augment 4); variants C and D skipped.")

    variants = {"A head only": (False, None), "B balanced weights": (True, None)}
    if aug is not None:
        variants["C augmented copies"] = (False, aug)
        variants["D augmented + weights"] = (True, aug)

    print(f"Pool: {len(y)} images in {len(np.unique(groups))} photograph groups; "
          f"{config.CV_FOLDS} folds x {config.CV_REPEATS} repeats per variant")
    t0 = time.time()
    proba, preds, per_repeat = {}, {}, {}
    for name, (weighted, a) in variants.items():
        fn = keras_predict_fn(X, y, n, weighted, a)
        proba[name] = run_cv(len(y), y, groups, fn, n, config.CV_FOLDS, config.CV_REPEATS, config.SEED)
        preds[name] = proba[name].mean(axis=0).argmax(axis=1)
        per_repeat[name] = [metrics(y, proba[name][r].argmax(axis=1), n)["macro_f1"]
                            for r in range(config.CV_REPEATS)]
        print(f"{name:24s} macro-F1 per repeat mean {np.mean(per_repeat[name]):.3f} "
              f"(sd {np.std(per_repeat[name]):.3f})  [{time.time() - t0:.0f} s]", flush=True)

    np.savez_compressed(config.CV_OOF_FILE, files=files, y=y, groups=groups, names=np.array(names),
                        **{k.split()[0]: proba[k] for k in variants})   # out-of-fold probabilities, (repeats, images, classes)
    boot = group_bootstrap(y, groups, preds, n, config.BOOTSTRAP_SAMPLES, config.SEED)
    ref = "A head only"
    chosen = ref
    for name in variants:
        if name != ref and prefer_variant(boot, ref, name) and \
                metrics(y, preds[name], n)["macro_f1"] > metrics(y, preds[chosen], n)["macro_f1"]:
            chosen = name

    L = ["# Cross-Validation Report (Stage 6a)\n",
         "Repeated group-stratified cross-validation of the frozen ResNet50 head. **Train and validation images only; "
         "the test set was not used.** All images of a photograph group are always on the same side of a fold.\n",
         "## Setup\n",
         f"- Pool: {len(y)} images in {len(np.unique(groups))} photograph groups; "
         f"{config.CV_FOLDS} folds, {config.CV_REPEATS} repeats with different group assignments, seed {config.SEED}",
         f"- Head trained for a fixed {config.CV_EPOCHS} epochs per fit (no early stopping: there is no spare validation data)",
         f"- Out-of-fold predictions are averaged over repeats. Intervals are 95 percent, from {config.BOOTSTRAP_SAMPLES} "
         "resamples of whole photograph groups within each class",
         "- Selection rule (fixed in advance): variant A is kept unless another variant improves macro-F1 and the 95 percent "
         "interval of the paired difference from A lies entirely above zero.\n",
         "| Class | Images | Photograph groups |", "|---|---|---|"]
    for c, nm in enumerate(names):
        L.append(f"| {nm} | {int((y == c).sum())} | {len(np.unique(groups[y == c]))} |")
    L += ["\n## Variants (out-of-fold)\n",
          "| Variant | Accuracy | Macro-F1 | Macro-F1 95% interval | Macro-F1 per repeat (mean, sd) | Difference from A (95% interval) |",
          "|---|---|---|---|---|---|"]
    for name in variants:
        m = metrics(y, preds[name], n)
        lo, hi = ci(boot[name]["macro_f1"])
        if name == ref:
            dtxt = "reference"
        else:
            dl, dh = ci(boot[name]["macro_f1"] - boot[ref]["macro_f1"])
            dtxt = f"{boot[name]['macro_f1'].mean() - boot[ref]['macro_f1'].mean():+.3f} ({dl:+.3f} to {dh:+.3f})"
        L.append(f"| {name}{' (chosen)' if name == chosen else ''} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | "
                 f"{lo:.3f} to {hi:.3f} | {np.mean(per_repeat[name]):.3f}, {np.std(per_repeat[name]):.3f} | {dtxt} |")
    L.append(f"\nChosen by the rule: **{chosen}**.\n")
    for name in variants:
        m = metrics(y, preds[name], n)
        gc = group_correctness(y, groups, preds[name], n)
        L += [f"## Per-class results: {name}\n",
              "| Class | Recall | Recall 95% interval | Precision | Photograph groups | Groups mostly correct |",
              "|---|---|---|---|---|---|"]
        for c, nm in enumerate(names):
            lo, hi = ci(boot[name]["recall"][:, c])
            L.append(f"| {nm} | {m['recall'][c]:.3f} | {lo:.3f} to {hi:.3f} | {m['precision'][c]:.3f} | {gc[c][0]} | {gc[c][1]} |")
        L += ["", "Confusion matrix (rows actual, columns predicted):\n",
              "| | " + " | ".join(names) + " |", "|---|" + "---|" * n]
        for i, nm in enumerate(names):
            L.append(f"| **{nm}** | " + " | ".join(str(v) for v in m["confusion"][i]) + " |")
        L.append("")
    L += ["## Reading this report\n",
          "- The intervals reflect the small number of photographs, not the number of images; they are wide.",
          "- Each class has between 4 and about 50 photograph groups, so a class result rests on very few independent photographs.",
          "- Cross-validation scores describe this dataset only. They say nothing about other photographers, cameras or regions."]
    config.CV_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    config.CV_JSON.write_text(json.dumps({
        "pool_images": int(len(y)), "groups": int(len(np.unique(groups))), "folds": config.CV_FOLDS,
        "repeats": config.CV_REPEATS, "chosen": chosen,
        "variants": {k: {"macro_f1": metrics(y, preds[k], n)["macro_f1"], "accuracy": metrics(y, preds[k], n)["accuracy"],
                         "macro_f1_ci95": ci(boot[k]["macro_f1"])} for k in variants}}, indent=2), encoding="utf-8")
    print(f"Chosen: {chosen}. Wrote {config.CV_REPORT} and {config.CV_JSON}. Total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
