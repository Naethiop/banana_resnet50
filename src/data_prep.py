"""Stage 4: cleaning, label conversion and group-aware splitting.

Run from the project root, after scripts/audit_dataset.py:
    python -m src.data_prep
    python -m src.data_prep --force        # rebuild data/processed

What it does (approved decisions):
  1. Reads reports/dataset_audit.csv (class labels and file-name/folder conflict flags).
  2. Excludes every image in a near-duplicate group (similarity >= CONFLICT_THRESHOLD, across all
     classes) that carries conflicting class labels or contains an image whose file name disagrees
     with its folder, and lists them in reports/excluded_images.csv.
  3. Splits the remaining PHOTOGRAPH GROUPS, class by class, into train/val/test with a fixed
     seed, so that no photograph (and none of its near-duplicate copies) is in two splits.
     Groups for splitting are formed WITHIN each class: images are linked when their pixel similarity
     is >= GROUP_THRESHOLD or their ResNet50-feature cosine similarity is >= EMBEDDING_LINK_THRESHOLD
     (the second catches crops, zoom and rotations; it needs data/features from src.features).
     Comparing within a class avoids linking look-alike leaves of different classes. Group counts follow 70/20/10 per class; among seeded random assignments, the one whose
     image shares are closest to 70/20/10 is kept. Only group sizes are used, never image content
     or any model result.
  4. Copies the images into data/processed/<split>/<class>/ and writes the assignment table,
     class names and a split report, including a residual-leakage check.
The raw export is never modified.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.dedup import similarity_from_paths, union_find_groups  # noqa: E402
from src.features import features_for_files  # noqa: E402
from src.leakage_embeddings import cosine_matrix  # noqa: E402

SPLITS = ("train", "val", "test")


def _round_half_up(x):
    return int(x + 0.5)


def assign_splits(df, ratios=None, seed=config.SEED, tries=config.SPLIT_SEARCH_TRIES,
                  min_groups=config.MIN_GROUPS_PER_SPLIT):
    """Return {group_id: split} using whole groups, stratified by class.

    df needs columns: label, group_id, file. The result does not depend on the numbering of
    group_id or on row order: groups are put in a canonical order (smallest file name) first.
    """
    ratios = ratios or config.SPLIT_RATIOS
    rng = np.random.default_rng(seed)
    out = {}
    for label in sorted(df.label.unique()):
        sub = df[df.label == label]
        info = sub.groupby("group_id").agg(size=("file", "count"), key=("file", "min"))
        info = info.sort_values("key")
        gids = list(info.index)
        sizes = info["size"].to_numpy()
        n = len(gids)
        # classes with very few photograph groups cannot give the full minimum to every split
        min_eff = max(1, min(min_groups, n // 3))
        n_test = max(min_eff, _round_half_up(ratios["test"] * n))
        n_val = max(min_eff, _round_half_up(ratios["val"] * n))
        if n - n_test - n_val < min_eff:
            raise ValueError(f"Class {label}: only {n} photograph groups, too few for three splits")
        total = sizes.sum()
        best_cost, best_perm = None, None
        for _ in range(tries):
            perm = rng.permutation(n)
            shares = {
                "test": sizes[perm[:n_test]].sum() / total,
                "val": sizes[perm[n_test:n_test + n_val]].sum() / total,
            }
            shares["train"] = 1.0 - shares["test"] - shares["val"]
            cost = sum(abs(shares[k] - ratios[k]) for k in SPLITS)
            if best_cost is None or cost < best_cost:
                best_cost, best_perm = cost, perm
        for pos, idx in enumerate(best_perm):
            split = "test" if pos < n_test else "val" if pos < n_test + n_val else "train"
            out[gids[idx]] = split
    return out


def load_audit(audit_csv):
    df = pd.read_csv(audit_csv)
    needed = {"split", "label", "file", "relpath", "group_id", "needs_review",
              "in_label_conflict_group", "hint_conflict"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"{audit_csv} lacks columns {sorted(missing)}. Re-run scripts/audit_dataset.py.")
    df["relpath"] = df["relpath"].str.replace("\\", "/", regex=False)
    return df


def exclusions(df):
    """df has a group_id column. Flag whole groups with conflicting labels or a file-name conflict."""
    n_labels = df.groupby("group_id").label.nunique()
    conflict_groups = set(n_labels[n_labels > 1].index)
    hint_groups = set(df[df.hint_conflict].group_id)
    flagged = conflict_groups | hint_groups
    ex = df[df.group_id.isin(flagged)].copy()
    ex["group_id"] = ex["group_id"]  # numbering here is the conflict-detection grouping

    def reason(r):
        parts = []
        if r.group_id in conflict_groups:
            parts.append("near-duplicate group carries conflicting class labels")
        if r.hint_conflict:
            parts.append("file name disagrees with folder label")
        if not parts:
            parts.append("same photograph group as a flagged image")
        return "; ".join(parts)

    ex["reason"] = ex.apply(reason, axis=1)
    ex["in_stage3_list_of_73"] = ex.needs_review
    ex = ex.rename(columns={"group_id": "conflict_group_id"})
    return ex


def leakage_check(kept, sim_kept):
    """Residual cross-split similarity between kept images at the configured thresholds."""
    split = kept.split_new.to_numpy()
    rows = []
    for thr in config.LEAK_CHECK_THRESHOLDS:
        ii, jj = np.where(np.triu(sim_kept >= thr, k=1))
        cross = [(i, j) for i, j in zip(ii, jj) if split[i] != split[j]]
        test_imgs = {i for i, j in cross if split[i] == "test"} | {j for i, j in cross if split[j] == "test"}
        rows.append((thr, len(cross), len(test_imgs)))
    return rows


def embedding_leakage(kept, emb_same):
    split = kept.split_new.to_numpy()
    rows = []
    for thr in config.EMBEDDING_LEAK_THRESHOLDS:
        ii, jj = np.where(np.triu(emb_same >= thr, k=1))
        cross = [(i, j) for i, j in zip(ii, jj) if split[i] != split[j]]
        test_imgs = {i for i, j in cross if split[i] == "test"} | {j for i, j in cross if split[j] == "test"}
        rows.append((thr, len(cross), len(test_imgs)))
    return rows


def write_report(kept, ex, leak, leak_emb, seed, ratios, threshold, conflict_threshold, emb_threshold):
    n_kept, n_ex = len(kept), len(ex)
    L = ["# Split Report (Stage 4)\n",
         "Generated by `src/data_prep.py`. All figures are measured from the prepared data.\n",
         "## Parameters\n",
         f"- Seed: {seed}",
         f"- Target ratios (approved): train {ratios['train']:.0%}, val {ratios['val']:.0%}, test {ratios['test']:.0%}",
         f"- Exclusion rule: near-duplicate groups (similarity >= {conflict_threshold}, all classes) with conflicting labels or a file-name/folder conflict",
         f"- Unit of splitting: photograph group, formed within each class; linked when pixel similarity >= {threshold} (brightness-, flip- and rotation-invariant)"
         + (f" or ResNet50-feature cosine similarity >= {emb_threshold}" if emb_threshold > 0 else " (embedding link disabled)"),
         f"- Minimum groups per class in each split: {config.MIN_GROUPS_PER_SPLIT}",
         f"- Images kept: {n_kept}; excluded: {n_ex}, of which {int(ex.in_stage3_list_of_73.sum())} were in the Stage 3 list of 73 (see `reports/excluded_images.csv`)\n",
         "## Images per class and split\n",
         "| Class | train | val | test | total |", "|---|---|---|---|---|"]
    ct = pd.crosstab(kept.label, kept.split_new).reindex(columns=list(SPLITS), fill_value=0)
    for lab, r in ct.iterrows():
        L.append(f"| {lab} | {r.train} | {r.val} | {r.test} | {r.sum()} |")
    t = ct.sum()
    L.append(f"| **All** | {t.train} | {t.val} | {t.test} | {t.sum()} |")
    L.append(f"\nImage shares: train {t.train/t.sum():.1%}, val {t.val/t.sum():.1%}, test {t.test/t.sum():.1%}.\n")
    L.append("## Distinct photograph groups per class and split\n")
    L.append("| Class | train | val | test | total |\n|---|---|---|---|---|")
    gt = kept.groupby(["label", "split_new"]).group_id.nunique().unstack(fill_value=0).reindex(columns=list(SPLITS), fill_value=0)
    for lab, r in gt.iterrows():
        L.append(f"| {lab} | {r.train} | {r.val} | {r.test} | {r.sum()} |")
    g = gt.sum()
    L.append(f"| **All** | {g.train} | {g.val} | {g.test} | {g.sum()} |")
    L.append("\nEvery metric reported for a class rests on the number of distinct photographs in its test "
             "column above, not on the number of images.\n")
    L.append("## Residual leakage check\n")
    L.append(f"Same-class image pairs in different splits that are still similar. At the grouping threshold "
             f"({threshold}) there should be none. Pairs at lower thresholds may be real copies that the "
             f"grouping missed (for example crops or zooms), or only look-alike photographs.\n")
    L.append("| Similarity threshold | Cross-split image pairs | Test images involved |\n|---|---|---|")
    for thr, pairs, timgs in leak:
        L.append(f"| {thr} | {pairs} | {timgs} |")
    if leak_emb:
        L.append("\nSame-class pairs in different splits by ResNet50-feature cosine similarity:\n")
        L.append("| Cosine threshold | Cross-split image pairs | Test images involved |\n|---|---|---|")
        for thr, pairs, timgs in leak_emb:
            L.append(f"| {thr} | {pairs} | {timgs} |")
    few = [lab for lab, r in gt.iterrows() if r.sum() < 6]
    if few:
        L.append("\nWARNING: " + ", ".join(few) + " has fewer than 6 photograph groups, so its validation and test "
                 "sets are single photographs. Results for such classes are anecdotal.")
    L.append("\n## Notes\n")
    L.append("- The split used only class labels, group membership and group sizes. No image content "
             "beyond the near-duplicate grouping and no model result was used.")
    L.append("- The test set must not be used for model selection or tuning (Stages 5 to 7).")
    L.append("- All data come from one source (the Roboflow export). No external test set exists; "
             "generalisation to other photographers, devices and regions is untested.")
    config.SPLIT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=str(config.EXPORT_DIR))
    ap.add_argument("--audit-csv", default=str(config.AUDIT_CSV))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--conflict-threshold", type=float, default=config.CONFLICT_THRESHOLD)
    ap.add_argument("--group-threshold", type=float, default=config.GROUP_THRESHOLD)
    ap.add_argument("--embedding-threshold", type=float, default=config.EMBEDDING_LINK_THRESHOLD,
                    help="0 disables the embedding link (pixel similarity only)")
    ap.add_argument("--force", action="store_true", help="rebuild data/processed if it exists")
    args = ap.parse_args()

    export_dir = Path(args.export)
    df = load_audit(args.audit_csv)
    missing = [p for p in df.relpath if not (export_dir / p).is_file()]
    if missing:
        sys.exit(f"{len(missing)} files listed in the audit are missing from {export_dir}, e.g. {missing[:3]}")

    print("Comparing photographs (about a minute)...")
    sim = similarity_from_paths([export_dir / p for p in df.relpath])
    df["group_id"] = union_find_groups(sim, args.conflict_threshold)      # used only to find conflicts

    ex = exclusions(df)
    keep_mask = ~df.group_id.isin(set(df[df.index.isin(ex.index)].group_id))
    kept = df[keep_mask].copy().reset_index(drop=True)
    idx = np.where(keep_mask)[0]
    sim_kept = sim[np.ix_(idx, idx)]
    # same-class similarity only: different-class pairs are made non-linking
    same = kept.label.to_numpy()[:, None] == kept.label.to_numpy()[None, :]
    sim_pix_same = np.where(same, sim_kept, -1.0)
    emb_same = None
    sim_link = sim_pix_same
    if args.embedding_threshold > 0:
        if not config.FEATURES_FILE.exists():
            sys.exit("Embedding features are needed for grouping. Run python -m src.features first "
                     "(or use --embedding-threshold 0 for pixel similarity only).")
        emb = cosine_matrix(features_for_files(kept.file.tolist()))
        emb_same = np.where(same, emb, -1.0)
        sim_link = np.where(emb_same >= args.embedding_threshold, 1.0, sim_pix_same)
    kept["group_id"] = union_find_groups(sim_link, args.group_threshold)   # photograph groups for splitting
    print(f"Conflict threshold {args.conflict_threshold}: images {len(df)}; excluded {len(ex)} in "
          f"{ex.conflict_group_id.nunique()} groups; kept {len(kept)}")
    print(f"Group threshold {args.group_threshold} (within class): {kept.group_id.nunique()} photograph groups")

    assignment = assign_splits(kept, seed=args.seed)
    kept["split_new"] = kept.group_id.map(assignment)

    processed = config.DATA_PROCESSED
    existing = [processed / s for s in SPLITS if (processed / s).exists()]
    if existing and not args.force:
        sys.exit("data/processed already holds splits. Use --force to rebuild them.")
    for d in existing:
        shutil.rmtree(d)
    for _, r in kept.iterrows():
        dest = processed / r.split_new / r.label
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(export_dir / r.relpath, dest / r.file)

    config.DATA_SPLITS.mkdir(parents=True, exist_ok=True)
    out = kept[["file", "label", "group_id", "split_new", "split", "relpath"]].rename(
        columns={"split_new": "assigned_split", "split": "original_export_split"})
    out.sort_values(["assigned_split", "label", "file"]).to_csv(config.SPLIT_CSV, index=False)
    ex[["file", "label", "conflict_group_id", "split", "relpath", "reason", "in_stage3_list_of_73"]].sort_values(["conflict_group_id", "file"]).to_csv(
        config.EXCLUDED_CSV, index=False)
    class_names = sorted(kept.label.unique())
    config.CLASS_NAMES_SPLIT_FILE.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    leak = leakage_check(kept, sim_pix_same)
    leak_emb = embedding_leakage(kept, emb_same) if emb_same is not None else []
    write_report(kept, ex, leak, leak_emb, args.seed, config.SPLIT_RATIOS, args.group_threshold,
                 args.conflict_threshold, args.embedding_threshold)
    print("Done. Wrote data/processed, data/splits/split_assignments.csv, data/splits/class_names.json, "
          "reports/excluded_images.csv and reports/split_report.md")
    print(pd.crosstab(kept.label, kept.split_new).reindex(columns=list(SPLITS), fill_value=0).to_string())
    print("Residual pixel similarity (threshold, cross-split pairs, test images):", leak)
    print("Residual embedding similarity (threshold, cross-split pairs, test images):", leak_emb)


if __name__ == "__main__":
    main()
