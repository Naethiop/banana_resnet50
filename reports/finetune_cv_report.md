# Fine-Tuning Report (Stage 6b)

Two-stage training compared with the head-only model on the SAME group cross-validation folds (one repeat, 4 folds). **Train and validation images only; the test set was not used.**

## Setup

- Stage 1: head on cached features plus 4 augmented copies, balanced class weights (variant D)
- Stage 2: layers named `conv5_*` unfrozen (batch-norm layers frozen), Adam learning rate 0.0001, 6 fixed epochs, fresh augmentation every epoch, batch size 16
- Seed 42; one repeat of the cross-validation, so fold assignment variability is not averaged out
- Total run time 39 minutes
- Intervals: 95 percent, 2000 resamples of whole photograph groups within each class, same resamples for both models

## Comparison

| Model | Accuracy | Macro-F1 | Macro-F1 95% interval |
|---|---|---|---|
| head only (stage 1) | 0.697 | 0.684 | 0.581 to 0.834 |
| fine-tuned (stage 2) | 0.691 | 0.643 | 0.534 to 0.827 |

Paired macro-F1 difference (fine-tuned minus head only): -0.038 (95% interval -0.081 to +0.011).

Selection rule (as in Stage 6a): fine-tuning is justified only if this interval lies entirely above zero. **Result: not justified; keep the head-only model.**

## Per-class results: head only (stage 1)

| Class | Recall | Recall 95% interval | Precision | Photograph groups | Groups mostly correct |
|---|---|---|---|---|---|
| Black-Sigatoka | 0.598 | 0.326 to 0.882 | 0.562 | 20 | 13 |
| Healthy-Leaf | 1.000 | 1.000 to 1.000 | 0.942 | 48 | 48 |
| Panama-Disease | 0.263 | 0.138 to 1.000 | 0.955 | 4 | 2 |
| Potassium-Deficiency | 0.785 | 0.563 to 0.965 | 0.680 | 13 | 11 |
| Yellow-Sigatoka | 0.740 | 0.497 to 0.890 | 0.722 | 15 | 9 |

Confusion matrix (rows actual, columns predicted):

| | Black-Sigatoka | Healthy-Leaf | Panama-Disease | Potassium-Deficiency | Yellow-Sigatoka |
|---|---|---|---|---|---|
| **Black-Sigatoka** | 113 | 4 | 0 | 45 | 27 |
| **Healthy-Leaf** | 0 | 81 | 0 | 0 | 0 |
| **Panama-Disease** | 2 | 0 | 21 | 34 | 23 |
| **Potassium-Deficiency** | 43 | 0 | 0 | 183 | 7 |
| **Yellow-Sigatoka** | 43 | 1 | 1 | 7 | 148 |

## Per-class results: fine-tuned (stage 2)

| Class | Recall | Recall 95% interval | Precision | Photograph groups | Groups mostly correct |
|---|---|---|---|---|---|
| Black-Sigatoka | 0.640 | 0.361 to 0.928 | 0.590 | 20 | 16 |
| Healthy-Leaf | 0.975 | 0.926 to 1.000 | 0.840 | 48 | 48 |
| Panama-Disease | 0.138 | 0.014 to 1.000 | 0.846 | 4 | 2 |
| Potassium-Deficiency | 0.824 | 0.592 to 0.990 | 0.649 | 13 | 11 |
| Yellow-Sigatoka | 0.690 | 0.438 to 0.859 | 0.789 | 15 | 10 |

Confusion matrix (rows actual, columns predicted):

| | Black-Sigatoka | Healthy-Leaf | Panama-Disease | Potassium-Deficiency | Yellow-Sigatoka |
|---|---|---|---|---|---|
| **Black-Sigatoka** | 121 | 0 | 0 | 53 | 15 |
| **Healthy-Leaf** | 0 | 79 | 0 | 0 | 2 |
| **Panama-Disease** | 0 | 5 | 11 | 44 | 20 |
| **Potassium-Deficiency** | 37 | 3 | 1 | 192 | 0 |
| **Yellow-Sigatoka** | 47 | 7 | 1 | 7 | 138 |

## Notes

- The number of epochs was fixed in advance; it was not tuned on the test folds.
- Cross-validation scores describe this dataset only and rest on very few independent photographs.
- One repeat only: a different fold assignment could change the result.
