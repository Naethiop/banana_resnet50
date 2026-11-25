# Baseline Validation Report (Stage 5)

Frozen ResNet50 (ImageNet) base with a new softmax layer. **Validation data only. The test set has not been used.**

## Settings compared on validation data

| Class weights | Epochs run | Validation accuracy | Validation macro-F1 |
|---|---|---|---|
| none (chosen) | 15 | 0.809 | 0.710 |
| balanced | 15 | 0.781 | 0.673 |

The validation set has 183 images from 22 photograph groups, so every figure below is a rough indication, not a precise estimate.

## Per-class validation results (none)

| Class | Precision | Recall | F1 | Validation images | Photograph groups |
|---|---|---|---|---|---|
| Black-Sigatoka | 0.737 | 1.000 | 0.848 | 42 | 4 |
| Healthy-Leaf | 1.000 | 1.000 | 1.000 | 18 | 11 |
| Panama-Disease | 0.000 | 0.000 | 0.000 | 25 | 1 |
| Potassium-Deficiency | 1.000 | 1.000 | 1.000 | 53 | 3 |
| Yellow-Sigatoka | 0.636 | 0.778 | 0.700 | 45 | 3 |

## Confusion matrix (rows actual, columns predicted)

| | Black-Sigatoka | Healthy-Leaf | Panama-Disease | Potassium-Deficiency | Yellow-Sigatoka |
|---|---|---|---|---|---|
| **Black-Sigatoka** | 42 | 0 | 0 | 0 | 0 |
| **Healthy-Leaf** | 0 | 18 | 0 | 0 | 0 |
| **Panama-Disease** | 5 | 0 | 0 | 0 | 20 |
| **Potassium-Deficiency** | 0 | 0 | 0 | 53 | 0 |
| **Yellow-Sigatoka** | 10 | 0 | 0 | 0 | 35 |

Figures: `reports/baseline_curves.png`, `reports/baseline_confusion_val.png`.

## Notes
- Training accuracy is not evidence of generalisation; use the validation columns.
- Class weights were compared on validation macro-F1 only.
- Augmentation is not used in this baseline (features are cached); it is introduced in Stage 6.
