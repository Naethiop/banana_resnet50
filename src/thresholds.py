"""Stage 8: low-confidence threshold from out-of-fold predictions, and an honest test of non-banana rejection.

    python -m src.cv                                    # once, to save out-of-fold probabilities (about 5 minutes)
    python -m src.thresholds                            # low-confidence threshold only
    python -m src.thresholds --negatives data/negatives # also test whether non-banana images can be rejected
    python -m src.thresholds --negatives data/negatives --adopt   # write the chosen values into models/model_meta.json

Low-confidence threshold (FR7): chosen from the out-of-fold predictions of the final method (Stage 6a, variant D;
5 repeats), never from the test set. Rule: the smallest threshold at which the predictions at or above it reach
config.TARGET_ACCURACY out-of-fold accuracy while at least config.MIN_COVERAGE of predictions are accepted; if no
threshold does, the largest threshold that still accepts config.MIN_COVERAGE of predictions.

Input check for non-banana images: two scores are compared on genuine banana images (out-of-fold) against the
supplied non-banana images: the largest softmax output, and the largest cosine similarity of the image's ResNet50
features to the training images. The rejection threshold is set from banana images only (at most
config.INPUT_CHECK_BANANA_REJECT of them rejected); the non-banana images are used only to measure the result.
A check is adopted only if its AUROC >= config.INPUT_CHECK_MIN_AUROC and it rejects >=
config.INPUT_CHECK_MIN_REJECTION of the non-banana images (criteria fixed before seeing results). Otherwise the
limitation is stated and the interface shows a warning.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from src import config

IMAGE_EXT = {".jpg", ".jpeg", ".png"}


# ---------- pure helpers (tested without TensorFlow) ----------
def selective_point(conf, correct, tau):
    accepted = conf >= tau
    cov = float(accepted.mean())
    acc = float(correct[accepted].mean()) if accepted.any() else float("nan")
    return cov, acc


def choose_threshold(conf, correct, target, min_cov, taus=None):
    taus = np.round(np.arange(0.30, 1.0, 0.01), 2) if taus is None else taus
    pts = [(t, *selective_point(conf, correct, t)) for t in taus]
    ok = [t for t, cov, acc in pts if cov >= min_cov and acc >= target]
    if ok:
        return float(min(ok)), "target reached"
    cand = [t for t, cov, acc in pts if cov >= min_cov]
    return (float(max(cand)) if cand else float(taus[0])), "target not reachable; largest threshold keeping the minimum coverage"


def bootstrap_selective(y, groups, conf, correct, tau, n_boot, seed):
    """conf, correct have shape (repeats, images). Resamples whole photograph groups within each class."""
    rng = np.random.default_rng(seed)
    members = {g: np.where(groups == g)[0] for g in np.unique(groups)}
    by_class = {c: np.unique(groups[y == c]) for c in np.unique(y)}
    cov, acc = np.zeros(n_boot), np.zeros(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[g] for c in by_class
                              for g in rng.choice(by_class[c], size=len(by_class[c]), replace=True)])
        cov[b], acc[b] = selective_point(conf[:, idx].ravel(), correct[:, idx].ravel(), tau)
    return cov, acc


def auroc(pos, neg):
    """Probability that a random positive scores higher than a random negative (ties count one half)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def knn_oof_similarity(X, folds):
    """For each held-out image: largest cosine similarity to the training-fold images (any class)."""
    Z = l2(X)
    out = np.zeros(len(X))
    for tr, te in folds:
        out[te] = (Z[te] @ Z[tr].T).max(axis=1)
    return out


def negative_source(path):
    """Source-image name for exports such as tile_21_png.rf.<hash>.jpg (several crops share one source)."""
    m = re.match(r"(.+?)_(?:png|jpe?g)\.rf\.", Path(path).name, flags=re.I)
    return m.group(1) if m else Path(path).stem


def find_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXT)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", help="folder of non-banana images (searched recursively)")
    ap.add_argument("--adopt", action="store_true",
                    help="write the low-confidence threshold into models/model_meta.json")
    ap.add_argument("--adopt-input-check", action="store_true",
                    help="also adopt the similarity input check (only if it met the criteria); implies --adopt")
    args = ap.parse_args()
    if not config.CV_OOF_FILE.exists():
        raise SystemExit("Run python -m src.cv first (it now saves out-of-fold probabilities).")
    from src.cv import load_pool, make_folds

    z = np.load(config.CV_OOF_FILE, allow_pickle=False)
    y, groups, names = z["y"], z["groups"], [str(n) for n in z["names"]]
    proba = z["D"]                                           # (repeats, images, classes)
    conf = proba.max(axis=2)
    correct = (proba.argmax(axis=2) == y[None, :])
    tau, how = choose_threshold(conf.ravel(), correct.ravel(), config.TARGET_ACCURACY, config.MIN_COVERAGE)
    cov, acc = selective_point(conf.ravel(), correct.ravel(), tau)
    bc, ba = bootstrap_selective(y, groups, conf, correct, tau, config.BOOTSTRAP_SAMPLES, config.SEED)
    clo, chi = np.percentile(bc, [2.5, 97.5])
    alo, ahi = np.nanpercentile(ba, [2.5, 97.5])

    L = ["# Threshold and Input-Check Report (Stage 8)\n",
         "Everything here uses out-of-fold predictions on the train and validation pool (final method, 5 repeats). "
         "**The test set was not used.**\n",
         "## Low-confidence threshold (FR7)\n",
         f"- Rule: smallest threshold with out-of-fold accuracy of accepted predictions >= {config.TARGET_ACCURACY} and "
         f"coverage >= {config.MIN_COVERAGE}; otherwise the largest threshold keeping coverage >= {config.MIN_COVERAGE}.",
         f"- Overall out-of-fold accuracy (all predictions): {correct.mean():.3f}; AUROC of confidence for separating correct "
         f"from wrong predictions: {auroc(conf[correct], conf[~correct]):.3f}",
         f"- **Chosen threshold: {tau:.2f}** ({how})",
         f"- At this threshold: coverage {cov:.3f} (95% interval {clo:.3f} to {chi:.3f}), accuracy of accepted predictions "
         f"{acc:.3f} (95% interval {alo:.3f} to {ahi:.3f}); intervals from resampling photograph groups\n",
         "| Threshold | Share of predictions accepted | Accuracy of accepted predictions |", "|---|---|---|"]
    for t in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99):
        c, a = selective_point(conf.ravel(), correct.ravel(), t)
        L.append(f"| {t:.2f} | {c:.3f} | {a:.3f} |")
    L += ["\nA warning below the threshold advises retaking the image. It does not prove that a prediction above the "
          "threshold is correct, and softmax confidence is not a calibrated probability.\n"]

    adopt_check = None
    if args.negatives:
        from src.predict import Predictor
        paths = find_images(args.negatives)
        if not paths:
            raise SystemExit(f"No jpg/png images found under {args.negatives}")
        pr = Predictor()
        P = load_pool()
        X = P["X"]
        folds = make_folds(P["y"], P["groups"], config.CV_FOLDS, config.SEED)          # repeat 0 folds
        knn_banana = knn_oof_similarity(X, folds)
        soft_banana = conf.ravel()
        bank = l2(X)
        soft_neg, knn_neg, pred_neg = [], [], []
        for p in paths:
            f, pb = pr.features_and_probs(p)
            soft_neg.append(float(pb.max()))
            pred_neg.append(pr.class_names[int(pb.argmax())])
            knn_neg.append(float((bank @ l2(f)).max()))
        soft_neg, knn_neg = np.array(soft_neg), np.array(knn_neg)
        L += [f"## Input check for non-banana images ({len(paths)} images from `{args.negatives}`)\n",
              "Check what the negative images show before interpreting the result. Aerial or satellite tiles, or other "
              "images unlike anything a user would photograph, are easy negatives: a pass would say nothing about leaves of other "
              "species, hands, soil, indoor objects or blurred banana photographs.\n",
              f"Thresholds are set from genuine banana images only, rejecting at most {config.INPUT_CHECK_BANANA_REJECT:.0%} of them. "
              f"Adoption criteria (fixed in advance): AUROC >= {config.INPUT_CHECK_MIN_AUROC} and at least "
              f"{config.INPUT_CHECK_MIN_REJECTION:.0%} of the non-banana images rejected.\n",
              "| Score | AUROC (banana vs non-banana) | Banana images rejected | Non-banana images rejected | 95% interval (Wilson) | Source images mostly rejected | 95% interval (Wilson) | Meets criteria |",
              "|---|---|---|---|---|---|---|---|"]
        sources = np.array([negative_source(p) for p in paths])
        uniq = sorted(set(sources))
        L.insert(len(L) - 3, f"The {len(paths)} images come from {len(uniq)} distinct source images (several crops share a source), "
                             "so the source-level figures are the fairer measure of certainty.\n")
        results = {}
        for name, bn, ng in (("largest softmax output", soft_banana, soft_neg),
                             ("largest cosine similarity to training images", knn_banana, knn_neg)):
            t = float(np.percentile(bn, 100 * config.INPUT_CHECK_BANANA_REJECT))
            rej_banana = float((bn < t).mean())
            k = int((ng < t).sum())
            lo, hi = wilson(k, len(ng))
            rej = ng < t
            ks = sum(1 for u in uniq if rej[sources == u].mean() > 0.5)
            slo, shi = wilson(ks, len(uniq))
            a = auroc(bn, ng)
            ok = a >= config.INPUT_CHECK_MIN_AUROC and k / len(ng) >= config.INPUT_CHECK_MIN_REJECTION
            results[name] = {"threshold": t, "auroc": a, "ok": ok}
            L.append(f"| {name} | {a:.3f} | {rej_banana:.1%} | {k} of {len(ng)} ({k / len(ng):.1%}) | {lo:.2f} to {hi:.2f} | "
                     f"{ks} of {len(uniq)} | {slo:.2f} to {shi:.2f} | {'yes' if ok else 'no'} |")
        import pandas as pd
        pd.DataFrame({
            "file": [str(Path(p).relative_to(args.negatives)) if str(p).startswith(str(args.negatives)) else str(p) for p in paths],
            "source": sources, "predicted_class": pred_neg, "max_softmax": np.round(soft_neg, 4),
            "max_similarity_to_training": np.round(knn_neg, 4),
            "rejected_by_softmax": soft_neg < results["largest softmax output"]["threshold"],
            "rejected_by_similarity": knn_neg < results["largest cosine similarity to training images"]["threshold"],
        }).to_csv(config.REPORTS_DIR / "negatives_scores.csv", index=False)
        L.append(f"Per-image scores: `reports/negatives_scores.csv`. Similarity threshold "
                 f"{results['largest cosine similarity to training images']['threshold']:.3f}.\n")
        knn_name = "largest cosine similarity to training images"
        if results[knn_name]["ok"]:
            adopt_check = {"type": "max cosine similarity of ResNet50 features to the training images",
                           "threshold": results[knn_name]["threshold"], "bank_images": int(len(X)),
                           "negatives_used_for_evaluation": len(paths)}
            L.append("\n**Result: the similarity-based input check meets the criteria on these negatives.** It is limited to "
                     "the kinds of image tested, which may not resemble what users upload.")
        else:
            L.append("\n**Result: no input check meets the criteria. The system cannot reliably reject unsuitable images. "
                     "State this limitation in Chapter Four and show an interface warning.**")
        L.append("")
    else:
        L += ["## Input check for non-banana images\n", "Not evaluated (no negative images supplied). State the limitation and show "
              "an interface warning.\n"]

    if args.adopt_input_check:
        args.adopt = True
        if adopt_check is None:
            raise SystemExit("The input check did not meet the criteria or no negatives were supplied; nothing adopted.")
    if args.adopt:
        if not args.adopt_input_check:
            adopt_check = None
        meta = json.loads(config.MODEL_META_FILE.read_text(encoding="utf-8"))
        meta["low_confidence_threshold"] = tau
        meta["threshold_selection"] = {"method": how, "target_accuracy": config.TARGET_ACCURACY,
                                       "min_coverage": config.MIN_COVERAGE, "oof_coverage": cov, "oof_accuracy": acc}
        meta["input_check"] = adopt_check
        if adopt_check:
            np.savez_compressed(config.INPUT_CHECK_BANK_FILE, bank=l2(load_pool()["X"]).astype("float32"))
        config.MODEL_META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        L.append(f"Adopted into `models/model_meta.json`: low-confidence threshold {tau:.2f}; input check "
                 f"{'adopted' if adopt_check else 'not adopted'}. The model weights are unchanged.")
    config.THRESHOLD_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
