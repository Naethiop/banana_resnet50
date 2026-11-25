# Stage 5 Notes: Preprocessing, Embedding Duplicate Check and Frozen-Base Baseline

## Design decisions (for review)
- **Preprocessing** (`src/preprocessing.py`): honour EXIF orientation, convert to RGB, centre-crop to a square, resize to 224 x 224, then the ResNet50 "caffe-mode" preprocessing (RGB to BGR, subtract ImageNet channel means), implemented in NumPy and tested against Keras. Dataset images are already square, so the crop only affects uploaded phone photographs. The alternative (squash a non-square photo to a square) distorts leaves; padding adds borders the model never saw. Centre-cropping can cut a lesion near the edge; this is recorded as a limitation and revisited in Stage 8.
- **Baseline**: frozen ResNet50 base plus a new softmax layer. The base is frozen, so its output per image never changes; features are computed once and only the new layer is trained (identical result, minutes instead of hours on a CPU). The trained layer is then assembled with the base into one `.keras` model and checked against the cached-feature predictions.
- **Augmentation** (flips, rotation within about +-14 degrees, brightness within +-20 percent) is defined in `src/train.py` for training data only and checked by a test. It is used from Stage 6, when images go through the network end to end. The baseline does not use it.
- **Class weights**: baseline trained twice (none, balanced by image count) and compared on VALIDATION macro-F1 only. The test set is never read for labels or metrics in this stage. Test features are extracted only so that the duplicate check can compare all splits; no labels or predictions are involved.
- **Embedding duplicate check** (`src/leakage_embeddings.py`): cosine similarity of ResNet50 features, less sensitive to crops and zoom than the pixel check. It reports only and never changes the split. It prints reference values (same-photograph pairs and different-class pairs) so that a threshold is judged, and saves a picture of cross-split pairs for inspection.

## Run order (activated environment, project root)
```
python -m pytest -q
python -m src.features
python -m src.leakage_embeddings
python -m src.train
python -m src.train --show-augmentation
```
- `src.features` downloads the ImageNet weights on first use (about 100 MB, internet needed) and prints the time per image. Please send me that timing.
- Expected files: `data/features/resnet50_gap.npz`, `reports/embedding_leakage_report.md`, `reports/embedding_cross_split_pairs.csv`, `reports/evidence/07_embedding_cross_split_pairs.png`, `reports/baseline_val_report.md`, `reports/baseline_curves.png`, `reports/baseline_confusion_val.png`, `reports/evidence/08_augmentation_examples.png`, `models/baseline_model.keras`, `models/baseline_meta.json`.

## Known limitations
- The assistant could not run TensorFlow code, so the Keras parts are untested until you run them. The NumPy and scikit-learn parts and the preprocessing are tested.
- The validation set has only about 37 photograph groups, so validation results are rough. Report them with the number of groups.
- Cached features mean the baseline sees no augmentation and no fine-tuning.
- Frozen ImageNet features may encode background and framing as well as symptoms.

## Review checkpoint 1 (after Stage 5)
Before Stage 6, review: the embedding duplicate report (does it show copies across splits?), the baseline validation report, the measured timings, and whether class weights help on validation.
