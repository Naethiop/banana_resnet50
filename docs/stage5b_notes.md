# Stage 5b Notes: Regrouping with Embedding Similarity (supersedes the Stage 4 grouping and result table)

## Why
The Stage 5 embedding check (ResNet50 features) showed same-class copies across splits that the pixel check had missed (crops, zoom, rotations). The student approved regrouping.

## Method
- Exclusions are unchanged: the same 73 images (conflict threshold 0.90 on pixel similarity).
- Within each class, two images are linked into one photograph group if pixel similarity >= 0.75 OR ResNet50-feature cosine similarity >= 0.85 (`config.EMBEDDING_LINK_THRESHOLD`).
- Choice of 0.85, judged from sample pairs (sampled by similarity band): copies were still common between 0.85 and 0.95 (Panama, Potassium deficiency, Yellow Sigatoka). Between 0.80 and 0.85 copies remained for Potassium deficiency but Yellow Sigatoka and Healthy linked different photographs; at 0.83 and below the Yellow Sigatoka groups merged into one of over 100 images. So 0.85 is the lowest value that did not merge unrelated photographs. Some copies below 0.85 can remain (see residual table).
- Cross-class pairs were checked: the highest-scoring Black Sigatoka / Yellow Sigatoka pairs are different leaves in similar scenes, not duplicates. No new exclusions.
- Features are now stored without split or group information. Split and group always come from `data/splits/split_assignments.csv`, so a re-split cannot leave stale information in the features file. The existing `data/features/resnet50_gap.npz` remains valid (same images).
- Classes with fewer than 6 groups get one group in each of validation and test (`min_eff = max(1, min(2, n // 3))`).

## Result of the run on the assistant's copy (the student's run should match)
| Class | train | val | test | total | groups train/val/test |
|---|---|---|---|---|---|
| Black-Sigatoka | 147 | 42 | 22 | 211 | 16 / 4 / 2 (22) |
| Healthy-Leaf | 63 | 18 | 9 | 90 | 37 / 11 / 5 (53) |
| Panama-Disease | 55 | 25 | 5 | 85 | 3 / 1 / 1 (5) |
| Potassium-Deficiency | 180 | 53 | 26 | 259 | 10 / 3 / 2 (15) |
| Yellow-Sigatoka | 155 | 45 | 20 | 220 | 12 / 3 / 2 (17) |
| **All** | 600 | 183 | 82 | 865 | 78 / 22 / 12 (112) |

Residual same-class similarity across splits: embedding 0.85 gives 0 pairs; 0.80 gives 152 pairs (20 test images); 0.75 gives 864 pairs. Pixel 0.75 gives 0 pairs; 0.70 gives 76 pairs (3 test images).

## What this means (state these limits in the thesis)
- At this level of duplicate control the 938 images are about 112 distinct photographs: Healthy about 53, Black Sigatoka 22, Yellow Sigatoka 17, Potassium deficiency 15, Panama 5.
- Validation and test sets are tiny: test holds 12 photographs in total, and Panama's validation and test sets are one photograph each. Per-class test results are anecdotal. Any figure must be reported with its photograph count.
- Previous validation results (Stage 5, original split) are inflated by copies. They are superseded and must not be cited as performance.
- Group-stratified cross-validation (approved) will give the main evidence. With 5 Panama groups, leave-one-group-out is the natural design for that class.
- The 0.80 residual means some copies of the same scene may still be in different splits, especially for Potassium deficiency.

## Run
```
python -m src.data_prep --force
python -m pytest -q
python -m src.leakage_embeddings
python -m src.train
```
`src.features` does not need to be rerun.
