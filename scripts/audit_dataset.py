"""Stage 3: dataset inspection audit for a Roboflow 'folder' classification export.

Usage (from the project root):
    python scripts/audit_dataset.py --data data/raw/<export_folder> \
        --class-map data/raw/class_map.json \
        --source-url https://universe.roboflow.com/rkn/banana-leaf-disease-detection-twfnp \
        --licence "CC BY 4.0"

The export folder must contain train/, valid/ (or val/) and test/ sub-folders,
each holding one folder per class.

Outputs (in reports/):
    dataset_audit.csv               one row per image with all flags
    dataset_inspection_report.md    facts measured from the files

What it checks: file integrity, format, resolution, exact duplicates, near-duplicate
photographs (brightness-, flip- and rotation-invariant), duplicate groups that span
splits (leakage), groups carrying conflicting class labels, folder/filename conflicts,
and class balance by image and by unique photograph.

It reports measurements only. Recommendations are made by a person.
Near-duplicate detection is a heuristic: it compares 48x48 greyscale versions after
removing brightness and contrast, under the 8 flips/rotations. It can miss heavily
cropped or rescaled copies, so the counts are minimums.
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dedup import similarity_from_thumbnails, thumbnail, union_find_groups  # noqa: E402
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "validation": "val", "test": "test"}
THRESHOLDS = (0.95, 0.90, 0.85)
MAIN_THRESHOLD = 0.90


def scan(data_dir, folder_map):
    rows = []
    for split_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        split = SPLIT_ALIASES.get(split_dir.name.lower())
        if split is None:
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() in IMAGE_EXT:
                    rows.append({
                        "split": split,
                        "folder": class_dir.name,
                        "label": folder_map.get(class_dir.name, class_dir.name),
                        "file": f.name,
                        "path": f,
                    })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="export folder holding train/valid/test")
    ap.add_argument("--class-map", help="JSON with 'folders' and optional 'filename_prefixes'")
    ap.add_argument("--source-url", default="(not provided)")
    ap.add_argument("--licence", default="(not provided)")
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_dir():
        sys.exit(f"Data folder not found: {data_dir}")
    cmap = json.loads(Path(args.class_map).read_text(encoding="utf-8")) if args.class_map else {}
    folder_map = cmap.get("folders", {})
    prefix_map = cmap.get("filename_prefixes", {})

    rows = scan(data_dir, folder_map)
    if not rows:
        sys.exit("No images found. Check the --data path.")
    n = len(rows)
    print(f"Found {n} images. Analysing...")

    readable = []
    vecs = []
    for r in rows:
        p = r["path"]
        r["bytes"] = p.stat().st_size
        r["md5"] = hashlib.md5(p.read_bytes()).hexdigest()
        r["error"] = ""
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.load()
                r["width"], r["height"] = im.size
                r["mode"], r["format"] = im.mode, im.format
                a = thumbnail(im)
            vecs.append(a)
            readable.append(True)
        except Exception as exc:  # noqa: BLE001 - record any failure
            r["error"] = f"{type(exc).__name__}: {exc}"
            r["width"] = r["height"] = 0
            r["mode"] = r["format"] = ""
            vecs.append(np.zeros((48, 48)))
            readable.append(False)

        m = re.match(r"([A-Za-z]+)_", r["file"])
        hint = prefix_map.get(m.group(1)) if m else None
        r["filename_hint"] = hint or ""
        r["hint_conflict"] = bool(hint and hint != r["label"])

    # near-duplicate similarity, invariant to brightness, flips and 90-degree rotations
    sim = similarity_from_thumbnails(vecs)

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "path"} for r in rows])
    df["relpath"] = [str(r["path"].relative_to(data_dir)) for r in rows]

    # exact duplicates
    df["exact_dup_count"] = df.groupby("md5")["md5"].transform("count")

    sens = []
    for thr in THRESHOLDS:
        gid = union_find_groups(sim, thr)
        d = df.assign(g=gid)
        sizes = d.groupby("g").size()
        multi = sizes[sizes > 1].index
        span = d[d.g.isin(multi)].groupby("g")["split"].nunique()
        conflict = d.groupby("g")["label"].nunique()
        in_cross = d.g.isin(span[span > 1].index)
        sens.append({
            "threshold": thr,
            "groups": int(sizes.size),
            "singleton_images": int((sizes == 1).sum()),
            "groups_with_2plus": int(len(multi)),
            "groups_spanning_splits": int((span > 1).sum()),
            "groups_with_label_conflict": int((conflict > 1).sum()),
            "images_in_cross_split_groups": int(in_cross.sum()),
        })
        if thr == MAIN_THRESHOLD:
            df["group_id"] = gid
    gsize = df.groupby("group_id")["group_id"].transform("count")
    df["group_size"] = gsize
    df["group_splits"] = df.groupby("group_id")["split"].transform("nunique")
    df["group_labels"] = df.groupby("group_id")["label"].transform("nunique")
    df["in_cross_split_group"] = (df.group_size > 1) & (df.group_splits > 1)
    df["in_label_conflict_group"] = df.group_labels > 1
    df["needs_review"] = df.in_label_conflict_group | df.hint_conflict

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    df.drop(columns=["bytes"]).assign(bytes=df["bytes"]).to_csv(out_dir / "dataset_audit.csv", index=False)

    # test / validation images with a near-duplicate in another split
    leak = {}
    for split in ("val", "test"):
        sub = df[df.split == split]
        other = df[df.split != split]
        gset = set(other.group_id)
        flag = sub.group_id.isin(gset) & (sub.group_size > 1)
        leak[split] = (int(flag.sum()), int(len(sub)))
    test_leak_by_class = {}
    for lab, sub in df[df.split == "test"].groupby("label"):
        flag = sub.group_id.isin(set(df[df.split != "test"].group_id)) & (sub.group_size > 1)
        test_leak_by_class[lab] = (int(flag.sum()), int(len(sub)))

    # ---- report ----
    L = []
    w = L.append
    w("# Dataset Inspection Report (Stage 3)\n")
    w("Generated by `scripts/audit_dataset.py`. Every figure below is measured from the downloaded files. "
      "Recommendations are not in this file's automatic part.\n")
    w("## 1. Source and licence\n")
    w(f"- Source recorded by the student: {args.source_url}")
    w(f"- Licence recorded by the student: {args.licence}")
    w(f"- Folder inspected: `{data_dir}`")
    readme = next(iter(data_dir.glob("README*")), None)
    if readme:
        txt = readme.read_text(encoding="utf-8", errors="ignore")
        keep = [ln.strip() for ln in txt.splitlines()
                if re.search(r"(images|annotated|pre-processing|augmentation|Auto-orientation|exported)", ln, re.I)]
        w("- Export notes found in the README:")
        for ln in keep:
            w(f"  - {ln}")
    w("")
    w("## 2. Images and classes\n")
    w(f"- Total images found: **{n}**")
    w(f"- Readable images: {sum(readable)}; unreadable or corrupt: {n - sum(readable)}")
    ct = pd.crosstab(df.label, df.split).reindex(columns=["train", "val", "test"], fill_value=0)
    ct = ct.astype(int)
    ct["total"] = ct.sum(axis=1)
    ct["share_%"] = (100 * ct.total / n).round(1)
    w("\n| Class | train | val | test | total | share % |\n|---|---|---|---|---|---|")
    for lab, r in ct.iterrows():
        w(f"| {lab} | {int(r.train)} | {int(r.val)} | {int(r.test)} | {int(r.total)} | {r['share_%']} |")
    tot = ct[["train", "val", "test"]].sum()
    w(f"| **All** | {tot.train} | {tot.val} | {tot.test} | {n} | 100 |")
    w(f"\nSplit proportions as supplied: train {100*tot.train/n:.1f}%, val {100*tot.val/n:.1f}%, test {100*tot.test/n:.1f}%.")
    w("Folder names in the export and the class each was mapped to:\n")
    w("| Folder name | Mapped class |\n|---|---|")
    for fo, lab in sorted(df.groupby("folder")["label"].first().items()):
        w(f"| `{fo}` | {lab} |")
    w("")
    w("## 3. Image quality\n")
    good = df[df.width > 0]
    fmts = {k: int(v) for k, v in good.format.value_counts().items()}
    modes = {k: int(v) for k, v in good["mode"].value_counts().items()}
    w(f"- Formats: {fmts}; colour modes: {modes}")
    size_counts = collections.Counter(zip(good.width, good.height))
    common = {f"{a}x{b}": c for (a, b), c in size_counts.most_common(3)}
    w(f"- Distinct image sizes: {len(size_counts)}; most common: {common}")
    w(f"- Images with a side below 224 px: {int(((good.width < 224) | (good.height < 224)).sum())}")
    w(f"- File size: min {df.bytes.min()//1024} KB, median {int(df.bytes.median())//1024} KB, max {df.bytes.max()//1024} KB")
    w("")
    w("## 4. Duplicates\n")
    w(f"- Exact duplicate files (identical bytes): {int((df.exact_dup_count > 1).sum())} images in "
      f"{df[df.exact_dup_count > 1].md5.nunique()} groups")
    w(f"- Near-duplicate analysis: images are compared after removing brightness and contrast, under mirror flips "
      f"and 90-degree rotations. The main threshold is {MAIN_THRESHOLD}. Sensitivity:\n")
    w("| Threshold | Groups | Singleton images | Groups with 2+ images | Groups spanning splits | Groups with label conflict | Images in cross-split groups |\n|---|---|---|---|---|---|---|")
    for s in sens:
        w(f"| {s['threshold']} | {s['groups']} | {s['singleton_images']} | {s['groups_with_2plus']} | "
          f"{s['groups_spanning_splits']} | {s['groups_with_label_conflict']} | {s['images_in_cross_split_groups']} |")
    w(f"\nAt the main threshold, the {n} images form **{df.group_id.nunique()} groups** (approximate number of distinct photographs).")
    w("")
    w("## 5. Leakage between the supplied splits\n")
    w(f"- Validation images with a near-duplicate in another split: {leak['val'][0]} of {leak['val'][1]}")
    w(f"- Test images with a near-duplicate in train or validation: **{leak['test'][0]} of {leak['test'][1]}**")
    w("\n| Class | Test images with a near-duplicate in train/val |\n|---|---|")
    for lab, (a, b) in sorted(test_leak_by_class.items()):
        w(f"| {lab} | {a} of {b} |")
    w("")
    w("## 6. Label consistency\n")
    w(f"- Images in a near-duplicate group that carries more than one class label: {int(df.in_label_conflict_group.sum())} "
      f"(in {df[df.in_label_conflict_group].group_id.nunique()} groups)")
    if prefix_map:
        w(f"- Images whose original file-name prefix disagrees with their folder label: {int(df.hint_conflict.sum())}")
        for _, r in df[df.hint_conflict].iterrows():
            w(f"  - `{r.relpath}`: folder says {r.label}, file name says {r.filename_hint}")
    w(f"- Images flagged for review by either test: {int(df.needs_review.sum())}")
    w("")
    w("## 7. Balance by image and by distinct photograph\n")
    per = df.groupby("label").agg(images=("file", "count"), groups=("group_id", "nunique"))
    w("| Class | Images | Approx. distinct photographs | Images per photograph |\n|---|---|---|---|")
    for lab, r in per.iterrows():
        w(f"| {lab} | {int(r.images)} | {int(r.groups)} | {r.images / r.groups:.1f} |")
    w("")
    w("## 8. Method limits\n")
    w("- Near-duplicate detection is a heuristic on 48x48 greyscale thumbnails. Heavily cropped, rescaled or "
      "recoloured copies can be missed, so duplicate counts are minimums.")
    w("- Single-linkage grouping can chain together different photographs that look alike; the sensitivity table "
      "shows how much the counts move with the threshold.")
    w("- Smooth, low-texture images (for example close-up leaf blades) give weak structure for the comparison, "
      "so duplicates among them may be under-detected.")
    w("- The script cannot judge whether a label is biologically correct. Flagged images need human review.")
    (out_dir / "dataset_inspection_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("Wrote reports/dataset_audit.csv and reports/dataset_inspection_report.md")
    print(f"Images {n}; unreadable {n - sum(readable)}; groups at {MAIN_THRESHOLD}: {df.group_id.nunique()}; "
          f"test images with partner in train/val: {leak['test'][0]} of {leak['test'][1]}")


if __name__ == "__main__":
    main()
