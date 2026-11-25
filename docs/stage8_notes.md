# Stage 8 Notes: Prediction Engine, Low-Confidence Threshold and Input Check

## Decisions (student)
- Model accepted as frozen: pooled final model, frozen ResNet50 + new head, augmentation, balanced weights, no fine-tuning.
- Non-banana test images: a Roboflow COCO export named "Images v1" (95 images). Audit by the assistant: the images are **aerial or satellite tiles** (ports, buildings, water, trees, vegetation; categories road, ship, buildings, trees, vegetation, vehicle, water, aeroplane), 1638 pixels wide, cropped from only **25 source tiles** (3 to 9 crops each). No banana content; no corrupt files; no exact duplicates. The export lists CC BY 4.0 but the provenance of the tiles is not recorded; record their real source before citing them.
- Consequence: these are easy, narrow negatives. They test only whether aerial imagery is rejected, not leaves of other plants, hands, soil, indoor objects or blurred banana photographs. A pass must not be reported as a general ability to reject non-banana images. Source-level figures (25 sources) are the fair measure of certainty.

## Design
- `src/predict.py`: loads the model package once, runs a self-check (split feature-extractor + head path equals the saved model), then classifies one image at a time. Returns label, confidence, top three, low-confidence flag, unsuitable flag, and the time taken. No Flask code.
- `src/thresholds.py`:
  - Low-confidence threshold chosen from the out-of-fold predictions of the final method (5 repeats), never from the test set. Rule fixed in advance: smallest threshold at which accepted predictions reach 0.85 out-of-fold accuracy with at least 50 percent of predictions accepted; otherwise the largest threshold keeping 50 percent coverage. Group-bootstrap intervals are reported.
  - Non-banana input check: two scores compared (largest softmax output; largest cosine similarity to the training images). The threshold is set from banana images only (at most 5 percent of banana images rejected); the object images only measure the result. Adopted only if AUROC >= 0.95 and at least 90 percent of the object images are rejected. Otherwise the limitation is stated and the interface shows a warning.
  - Limits of the evidence: the negatives are objects, not other plants. A pass would not show that leaves of other species are rejected.
- The adopted values go into `models/model_meta.json` only (weights unchanged).

## Run
```
python -m pytest -q
python -m src.cv                       # about 5 minutes; now also saves out-of-fold probabilities
python -m src.predict --benchmark
python -m src.thresholds --negatives data/negatives
```
Put the object images (any sub-folders) in `data/negatives/`. After reviewing `reports/threshold_report.md`, adopt with:
```
python -m src.thresholds --negatives data/negatives --adopt
```
