# Stage 4 Notes: Cleaning, Label Conversion and Group-Aware Split

## Decisions recorded
- Dataset: the Roboflow export only (student decision). No external test set exists, so generalisation to other photographers, cameras and regions is untested. State this as a limitation in Chapters Four and Five.
- Supplied train/validation/test split rejected (near-duplicate leakage). Group-aware re-split adopted.
- The 73 flagged images (9 near-duplicate groups with conflicting labels, or a file name that disagrees with the folder) are excluded and listed in `reports/excluded_images.csv`.
- Panama: option P1 (keep all Panama images; scope wording to be amended, limitation stated).
- Optional supplements approved: uncertainty reporting for per-class metrics and repeated group-stratified cross-validation (head-only model). Both are planned for Stages 6 and 7.

## Method
1. Class labels come from the confirmed folder mapping (`data/raw/class_map.json`). Class names: Black-Sigatoka, Healthy-Leaf, Panama-Disease, Potassium-Deficiency, Yellow-Sigatoka (written to `data/splits/class_names.json`).
2. Conflicts are found at similarity 0.90 across all classes (the Stage 3 rule). This reproduces the 73 exclusions.
3. Photograph groups for splitting are formed within each class at similarity 0.75. Reason: a check of the first version (0.90 everywhere) found same-class copies with different crops or zoom in different splits; a lower within-class threshold links more of them. Different-class pairs are never linked, because look-alike leaves of different classes at similarity 0.80 to 0.86 are not copies (checked visually).
4. Whole groups go to one split. Group counts follow 70/20/10 per class (at least 2 groups per class per split). Among 5000 seeded random assignments per class, the one whose image shares are closest to 70/20/10 is kept. Only group sizes are used.
5. Seed 42. The result does not depend on file order or group numbering.

## Result of the run on the assistant's copy (the student's run should match; report any difference)
| Class | train | val | test | total | groups (train/val/test) |
|---|---|---|---|---|---|
| Black-Sigatoka | 148 | 42 | 21 | 211 | 19 / 5 / 3 |
| Healthy-Leaf | 65 | 16 | 9 | 90 | 52 / 15 / 7 |
| Panama-Disease | 60 | 15 | 10 | 85 | 10 / 3 / 2 |
| Potassium-Deficiency | 181 | 52 | 26 | 259 | 21 / 6 / 3 |
| Yellow-Sigatoka | 155 | 45 | 20 | 220 | 28 / 8 / 4 |
| **All** | 609 | 170 | 86 | 865 | 130 / 37 / 19 |

Image shares 70.4 / 19.7 / 9.9 percent. 186 photograph groups. Same-class cross-split pairs still similar: none at 0.75, 21 pairs (5 test images) at 0.70, 128 pairs (8 test images) at 0.65.

## Limitations to state
- The test set holds only 19 photograph groups (Panama 2, Black Sigatoka 3, Potassium deficiency 3, Yellow Sigatoka 4, Healthy 7). Per-class test metrics are very coarse. Cross-validation and uncertainty intervals are the main evidence about generalisation; the single test result is a check.
- The duplicate detector compares small greyscale thumbnails. Crops, zooms and perspective changes can lower similarity, so some copies of one photograph may still lie in different splits (the residual table above). A second check using ResNet50 ImageNet features is planned for Stage 5, because it is less sensitive to crops and zoom.
- Group sizes differ greatly (the largest Potassium group holds dozens of images), so image-level metrics are dominated by a few photographs. Report photograph counts beside every metric.
- Only the near-duplicate and filename checks were used for labels. No expert has verified the labels.
- Panama images include pseudostem and leaf-sheath views (P1).
- Healthy images are mostly close-ups; the model may partly learn framing.
- All images come from one source.

## How to run
```
python -m src.data_prep --force
python -m pytest -q
```
Expected: 865 kept, 73 excluded, 186 photograph groups, and all tests passing. Open `reports/split_report.md`.
