"""Stage 7: descriptive diagnostics of the (already finished) test evaluation.

    python -m src.test_diagnostics

Reads reports/test_misclassified.csv (written by src.evaluate). It does NOT retrain, re-score or change the model.
For every test image it finds the most similar image of the same class in the train+validation pool (cosine
similarity of ResNet50 features) to show how much of the test score could come from near-copies of training
photographs. Disclose in the report that these diagnostics used the test labels after the evaluation.
"""
import numpy as np
import pandas as pd

from src import config
from src.features import load_features
from src.leakage_embeddings import cosine_matrix


def main():
    if not config.TEST_MISCLASSIFIED_CSV.exists():
        raise SystemExit("Run python -m src.evaluate first.")
    wrong_files = set(pd.read_csv(config.TEST_MISCLASSIFIED_CSV).file)
    d = load_features()
    feats, files, labels, splits, groups = (d[k] for k in ("features", "files", "labels", "splits", "groups"))
    sim = cosine_matrix(feats)
    test, pool = np.where(splits == "test")[0], np.where(splits != "test")[0]
    rows = []
    for i in test:
        same = pool[labels[pool] == labels[i]]
        rows.append((labels[i], int(groups[i]), float(sim[i, same].max()), files[i] in wrong_files))
    df = pd.DataFrame(rows, columns=["label", "group", "max_cos", "wrong"])
    L = ["# Test Evaluation Diagnostics (descriptive, after the single evaluation)\n",
         "These diagnostics used the test labels after the evaluation. They did not change the model.\n",
         "## Test images with a near-copy in the train and validation pool\n",
         "| Cosine threshold | Test images with a same-class pool image at or above it | of which misclassified |", "|---|---|---|"]
    for thr in (0.85, 0.80, 0.75):
        h = df[df.max_cos >= thr]
        L.append(f"| {thr} | {len(h)} of {len(df)} | {int(h.wrong.sum())} |")
    L += ["\n## Per class\n", "| Class | Test images | Photograph groups | Lowest | Median | Highest | Images at or above 0.80 | Misclassified |",
          "|---|---|---|---|---|---|---|---|"]
    for c, g in df.groupby("label"):
        L.append(f"| {c} | {len(g)} | {g.group.nunique()} | {g.max_cos.min():.2f} | {g.max_cos.median():.2f} | "
                 f"{g.max_cos.max():.2f} | {(g.max_cos >= 0.80).sum()} | {int(g.wrong.sum())} |")
    L += ["\n## Misclassified images by photograph group\n", "| Class | Group | Images in group (test) | Misclassified |", "|---|---|---|---|"]
    for (c, gid), g in df.groupby(["label", "group"]):
        if g.wrong.any():
            L.append(f"| {c} | {gid} | {len(g)} | {int(g.wrong.sum())} |")
    L += ["\n## Reading this\n",
          "- 'Highest similarity to a same-class pool image' of 0.80 and above, in earlier pair inspections, usually meant a crop, "
          "zoom or recolouring of the same scene. Test images with such partners may be easier than truly new photographs.",
          "- Classes whose test images have no such partners are a fairer check."]
    (config.REPORTS_DIR / "test_diagnostics.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[4:20]))


if __name__ == "__main__":
    main()
