# Threshold and Input-Check Report (Stage 8)

Everything here uses out-of-fold predictions on the train and validation pool (final method, 5 repeats). **The test set was not used.**

## Low-confidence threshold (FR7)

- Rule: smallest threshold with out-of-fold accuracy of accepted predictions >= 0.85 and coverage >= 0.5; otherwise the largest threshold keeping coverage >= 0.5.
- Overall out-of-fold accuracy (all predictions): 0.700; AUROC of confidence for separating correct from wrong predictions: 0.690
- **Chosen threshold: 0.94** (target not reachable; largest threshold keeping the minimum coverage)
- At this threshold: coverage 0.514 (95% interval 0.407 to 0.632), accuracy of accepted predictions 0.811 (95% interval 0.686 to 0.917); intervals from resampling photograph groups

| Threshold | Share of predictions accepted | Accuracy of accepted predictions |
|---|---|---|
| 0.50 | 0.971 | 0.704 |
| 0.60 | 0.903 | 0.713 |
| 0.70 | 0.821 | 0.733 |
| 0.80 | 0.725 | 0.752 |
| 0.90 | 0.594 | 0.786 |
| 0.95 | 0.489 | 0.817 |
| 0.99 | 0.328 | 0.893 |

A warning below the threshold advises retaking the image. It does not prove that a prediction above the threshold is correct, and softmax confidence is not a calibrated probability.

## Input check for non-banana images (118 images from `data/negatives`)

Check what the negative images show before interpreting the result. Aerial or satellite tiles, or other images unlike anything a user would photograph, are easy negatives: a pass would say nothing about leaves of other species, hands, soil, indoor objects or blurred banana photographs.

The 118 images come from 48 distinct source images (several crops share a source), so the source-level figures are the fairer measure of certainty.

Thresholds are set from genuine banana images only, rejecting at most 5% of them. Adoption criteria (fixed in advance): AUROC >= 0.95 and at least 90% of the non-banana images rejected.

| Score | AUROC (banana vs non-banana) | Banana images rejected | Non-banana images rejected | 95% interval (Wilson) | Source images mostly rejected | 95% interval (Wilson) | Meets criteria |
|---|---|---|---|---|---|---|---|
| largest softmax output | 0.528 | 5.0% | 6 of 118 (5.1%) | 0.02 to 0.11 | 0 of 48 | 0.00 to 0.07 | no |
| largest cosine similarity to training images | 0.984 | 5.1% | 111 of 118 (94.1%) | 0.88 to 0.97 | 41 of 48 | 0.73 to 0.93 | yes |
Per-image scores: `reports/negatives_scores.csv`. Similarity threshold 0.627.


**Result: the similarity-based input check meets the criteria on these negatives.** It is limited to the kinds of image tested, which may not resemble what users upload.

Adopted into `models/model_meta.json`: low-confidence threshold 0.94; input check adopted. The model weights are unchanged.
