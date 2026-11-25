# Cross-Validation Report (Stage 6a)

Repeated group-stratified cross-validation of the frozen ResNet50 head. **Train and validation images only; the test set was not used.** All images of a photograph group are always on the same side of a fold.

## Setup

- Pool: 783 images in 100 photograph groups; 4 folds, 5 repeats with different group assignments, seed 42
- Head trained for a fixed 15 epochs per fit (no early stopping: there is no spare validation data)
- Out-of-fold predictions are averaged over repeats. Intervals are 95 percent, from 2000 resamples of whole photograph groups within each class
- Selection rule (fixed in advance): variant A is kept unless another variant improves macro-F1 and the 95 percent interval of the paired difference from A lies entirely above zero.

| Class | Images | Photograph groups |
|---|---|---|
| Black-Sigatoka | 189 | 20 |
| Healthy-Leaf | 81 | 48 |
| Panama-Disease | 80 | 4 |
| Potassium-Deficiency | 233 | 13 |
| Yellow-Sigatoka | 200 | 15 |

## Variants (out-of-fold)

| Variant | Accuracy | Macro-F1 | Macro-F1 95% interval | Macro-F1 per repeat (mean, sd) | Difference from A (95% interval) |
|---|---|---|---|---|---|
| A head only | 0.672 | 0.634 | 0.535 to 0.810 | 0.629, 0.025 | reference |
| B balanced weights | 0.673 | 0.649 | 0.551 to 0.804 | 0.639, 0.023 | +0.014 (-0.010 to +0.034) |
| C augmented copies | 0.719 | 0.688 | 0.585 to 0.847 | 0.671, 0.023 | +0.052 (+0.010 to +0.094) |
| D augmented + weights (chosen) | 0.720 | 0.696 | 0.595 to 0.844 | 0.685, 0.023 | +0.058 (+0.012 to +0.104) |

Chosen by the rule: **D augmented + weights**.

## Per-class results: A head only

| Class | Recall | Recall 95% interval | Precision | Photograph groups | Groups mostly correct |
|---|---|---|---|---|---|
| Black-Sigatoka | 0.561 | 0.356 to 0.786 | 0.606 | 20 | 8 |
| Healthy-Leaf | 0.975 | 0.927 to 1.000 | 0.878 | 48 | 46 |
| Panama-Disease | 0.150 | 0.029 to 1.000 | 0.923 | 4 | 2 |
| Potassium-Deficiency | 0.682 | 0.481 to 0.938 | 0.665 | 13 | 10 |
| Yellow-Sigatoka | 0.850 | 0.669 to 0.960 | 0.639 | 15 | 12 |

Confusion matrix (rows actual, columns predicted):

| | Black-Sigatoka | Healthy-Leaf | Panama-Disease | Potassium-Deficiency | Yellow-Sigatoka |
|---|---|---|---|---|---|
| **Black-Sigatoka** | 106 | 11 | 0 | 33 | 39 |
| **Healthy-Leaf** | 1 | 79 | 0 | 1 | 0 |
| **Panama-Disease** | 7 | 0 | 12 | 40 | 21 |
| **Potassium-Deficiency** | 38 | 0 | 0 | 159 | 36 |
| **Yellow-Sigatoka** | 23 | 0 | 1 | 6 | 170 |

## Per-class results: B balanced weights

| Class | Recall | Recall 95% interval | Precision | Photograph groups | Groups mostly correct |
|---|---|---|---|---|---|
| Black-Sigatoka | 0.561 | 0.361 to 0.788 | 0.609 | 20 | 9 |
| Healthy-Leaf | 0.988 | 0.956 to 1.000 | 0.808 | 48 | 47 |
| Panama-Disease | 0.237 | 0.129 to 1.000 | 0.950 | 4 | 2 |
| Potassium-Deficiency | 0.652 | 0.432 to 0.938 | 0.691 | 13 | 10 |
| Yellow-Sigatoka | 0.850 | 0.669 to 0.960 | 0.630 | 15 | 12 |

Confusion matrix (rows actual, columns predicted):

| | Black-Sigatoka | Healthy-Leaf | Panama-Disease | Potassium-Deficiency | Yellow-Sigatoka |
|---|---|---|---|---|---|
| **Black-Sigatoka** | 106 | 18 | 0 | 26 | 39 |
| **Healthy-Leaf** | 1 | 80 | 0 | 0 | 0 |
| **Panama-Disease** | 4 | 0 | 19 | 36 | 21 |
| **Potassium-Deficiency** | 40 | 1 | 0 | 152 | 40 |
| **Yellow-Sigatoka** | 23 | 0 | 1 | 6 | 170 |

## Per-class results: C augmented copies

| Class | Recall | Recall 95% interval | Precision | Photograph groups | Groups mostly correct |
|---|---|---|---|---|---|
| Black-Sigatoka | 0.561 | 0.281 to 0.863 | 0.613 | 20 | 11 |
| Healthy-Leaf | 0.988 | 0.956 to 1.000 | 0.964 | 48 | 47 |
| Panama-Disease | 0.212 | 0.100 to 1.000 | 0.944 | 4 | 2 |
| Potassium-Deficiency | 0.828 | 0.638 to 0.979 | 0.687 | 13 | 11 |
| Yellow-Sigatoka | 0.835 | 0.644 to 0.950 | 0.732 | 15 | 12 |

Confusion matrix (rows actual, columns predicted):

| | Black-Sigatoka | Healthy-Leaf | Panama-Disease | Potassium-Deficiency | Yellow-Sigatoka |
|---|---|---|---|---|---|
| **Black-Sigatoka** | 106 | 3 | 0 | 46 | 34 |
| **Healthy-Leaf** | 1 | 80 | 0 | 0 | 0 |
| **Panama-Disease** | 3 | 0 | 17 | 36 | 24 |
| **Potassium-Deficiency** | 37 | 0 | 0 | 193 | 3 |
| **Yellow-Sigatoka** | 26 | 0 | 1 | 6 | 167 |

## Per-class results: D augmented + weights

| Class | Recall | Recall 95% interval | Precision | Photograph groups | Groups mostly correct |
|---|---|---|---|---|---|
| Black-Sigatoka | 0.571 | 0.303 to 0.863 | 0.610 | 20 | 11 |
| Healthy-Leaf | 0.988 | 0.956 to 1.000 | 0.941 | 48 | 47 |
| Panama-Disease | 0.250 | 0.100 to 1.000 | 0.952 | 4 | 2 |
| Potassium-Deficiency | 0.815 | 0.619 to 0.975 | 0.699 | 13 | 11 |
| Yellow-Sigatoka | 0.830 | 0.640 to 0.946 | 0.728 | 15 | 11 |

Confusion matrix (rows actual, columns predicted):

| | Black-Sigatoka | Healthy-Leaf | Panama-Disease | Potassium-Deficiency | Yellow-Sigatoka |
|---|---|---|---|---|---|
| **Black-Sigatoka** | 108 | 4 | 0 | 43 | 34 |
| **Healthy-Leaf** | 1 | 80 | 0 | 0 | 0 |
| **Panama-Disease** | 3 | 0 | 20 | 33 | 24 |
| **Potassium-Deficiency** | 39 | 0 | 0 | 190 | 4 |
| **Yellow-Sigatoka** | 26 | 1 | 1 | 6 | 166 |

## Reading this report

- The intervals reflect the small number of photographs, not the number of images; they are wide.
- Each class has between 4 and about 50 photograph groups, so a class result rests on very few independent photographs.
- Cross-validation scores describe this dataset only. They say nothing about other photographers, cameras or regions.
