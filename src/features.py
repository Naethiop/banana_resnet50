"""Stage 5: extract frozen ResNet50 (ImageNet) features for every kept image.

    python -m src.features            # about 5 to 15 minutes on a CPU-only laptop
    python -m src.features --force    # recompute

The features are the 2048-value global-average-pooled output of the ResNet50 base. They are used
(1) for the frozen-base baseline and (2) for the embedding-based duplicate check. Extracting
features for the test images does not use any labels or model results.
The first run downloads the ImageNet weights (about 100 MB) and needs an internet connection.
"""
import argparse
import time

import numpy as np
import pandas as pd

from src import config
from src.preprocessing import load_rgb, preprocess_batch


def load_split_table(csv=config.SPLIT_CSV):
    df = pd.read_csv(csv)
    df["path"] = [config.DATA_PROCESSED / s / lab / f
                  for s, lab, f in zip(df.assigned_split, df.label, df.file)]
    return df.sort_values(["assigned_split", "label", "file"]).reset_index(drop=True)


def features_for_files(files):
    """Cached features for the given file names, in the given order."""
    z = np.load(config.FEATURES_FILE, allow_pickle=False)
    pos = {str(f): i for i, f in enumerate(z["files"])}
    missing = [f for f in files if f not in pos]
    if missing:
        raise SystemExit(f"{len(missing)} images have no cached features, e.g. {missing[:2]}. "
                         "Run python -m src.features --force.")
    return z["features"][[pos[f] for f in files]]


def load_features(csv=config.SPLIT_CSV):
    """Features aligned to the CURRENT split table (sorted by split, class, file).

    Split names and group numbers always come from the split table, so they cannot go stale when the
    split is rebuilt; the features file only stores features, file names and class labels.
    """
    df = load_split_table(csv)
    return {"features": features_for_files(df.file.tolist()), "files": df.file.to_numpy(dtype=str),
            "labels": df.label.to_numpy(dtype=str), "groups": df.group_id.to_numpy(dtype=int),
            "splits": df.assigned_split.to_numpy(dtype=str), "table": df}


def extract(paths, batch_size=config.BATCH_SIZE):
    import keras

    base = keras.applications.ResNet50(include_top=False, weights="imagenet", pooling="avg",
                                       input_shape=tuple(config.IMAGE_SIZE) + (3,))
    feats, t0 = [], time.time()
    for start in range(0, len(paths), batch_size):
        batch = np.stack([load_rgb(p) for p in paths[start:start + batch_size]])
        feats.append(np.asarray(base.predict_on_batch(preprocess_batch(batch))))
        done = min(start + batch_size, len(paths))
        if (start // batch_size) % 5 == 0 or done == len(paths):
            print(f"  {done}/{len(paths)} images, {time.time() - t0:.0f} s", flush=True)
    return np.concatenate(feats).astype(np.float32), time.time() - t0


def extract_augmented(paths, copies, batch_size=config.BATCH_SIZE):
    """Features of `copies` randomly augmented versions of every image (training-style augmentation)."""
    import keras
    from src.train import build_augmentation

    keras.utils.set_random_seed(config.SEED)
    aug = build_augmentation()
    base = keras.applications.ResNet50(include_top=False, weights="imagenet", pooling="avg",
                                       input_shape=tuple(config.IMAGE_SIZE) + (3,))
    feats, src, t0 = [], [], time.time()
    for c in range(copies):
        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            batch = np.stack([load_rgb(p) for p in chunk])
            batch = np.clip(np.asarray(aug(batch, training=True)), 0, 255)
            feats.append(np.asarray(base.predict_on_batch(preprocess_batch(batch))))
            src.extend(p.name for p in chunk)
        print(f"  copy {c + 1}/{copies} done, {time.time() - t0:.0f} s", flush=True)
    return np.concatenate(feats).astype(np.float32), np.array(src, dtype=str), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--augment", type=int, default=0, metavar="K",
                    help="instead, cache features of K augmented copies of every train and val image")
    args = ap.parse_args()
    if args.augment:
        if not config.FEATURES_FILE.exists():
            raise SystemExit("Run python -m src.features first.")
        df = load_split_table()
        pool = df[df.assigned_split != "test"]          # the test images are never augmented or used
        print(f"Extracting {args.augment} augmented copies of {len(pool)} train/val images...")
        feats, src, secs = extract_augmented(list(pool.path), args.augment)
        np.savez_compressed(config.AUG_FEATURES_FILE, features=feats, src_files=src,
                            copies=args.augment, seed=config.SEED)
        print(f"Done in {secs / 60:.1f} minutes. Saved {feats.shape} to {config.AUG_FEATURES_FILE}")
        return
    if config.FEATURES_FILE.exists() and not args.force:
        raise SystemExit(f"{config.FEATURES_FILE} exists. Use --force to recompute.")
    if not config.SPLIT_CSV.exists():
        raise SystemExit("Run python -m src.data_prep first.")
    df = load_split_table()
    missing = [str(p) for p in df.path if not p.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} processed images missing, e.g. {missing[:2]}. Re-run data_prep.")
    print(f"Extracting features for {len(df)} images (batch size {config.BATCH_SIZE})...")
    feats, secs = extract(list(df.path))
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.FEATURES_FILE, features=feats, files=df.file.to_numpy(dtype=str),
        labels=df.label.to_numpy(dtype=str))
    print(f"Done in {secs / 60:.1f} minutes ({secs / len(df):.2f} s per image). "
          f"Saved {feats.shape} to {config.FEATURES_FILE}")


if __name__ == "__main__":
    main()
