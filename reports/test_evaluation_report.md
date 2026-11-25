# Final Test Evaluation (Stage 7)

The model was trained and saved before the test set was read, and the test set was evaluated once.

## Model

- Frozen ResNet50 (ImageNet weights) plus a new softmax layer; augmentation and balanced class weights; 15 epochs. Fine-tuning was not used (Stage 6b: not justified).
- Trained on: pool: 783 images, 100 photograph groups
- Test set: 82 images in 12 photograph groups, never used for training, model selection or threshold choice

## Test results

- Accuracy: **0.951**
- Macro-F1: **0.944** (95% group-bootstrap interval 0.899 to 1.000)
- Cross-validation estimate for the same method on the train and validation pool (Stage 6a, variant D): macro-F1 0.696, accuracy 0.720

| Class | Precision | Recall | Recall 95% interval | F1 | Test images | Photograph groups | Groups mostly correct |
|---|---|---|---|---|---|---|---|
| Black-Sigatoka | 1.000 | 0.818 | 0.810 to 1.000 | 0.900 | 22 | 2 | 2 |
| Healthy-Leaf | 0.692 | 1.000 | 1.000 to 1.000 | 0.818 | 9 | 5 | 5 |
| Panama-Disease | 1.000 | 1.000 | 1.000 to 1.000 | 1.000 | 5 | 1 | 1 |
| Potassium-Deficiency | 1.000 | 1.000 | 1.000 to 1.000 | 1.000 | 26 | 2 | 2 |
| Yellow-Sigatoka | 1.000 | 1.000 | 1.000 to 1.000 | 1.000 | 20 | 2 | 2 |

## Confusion matrix (rows actual, columns predicted)

| | Black-Sigatoka | Healthy-Leaf | Panama-Disease | Potassium-Deficiency | Yellow-Sigatoka |
|---|---|---|---|---|---|
| **Black-Sigatoka** | 18 | 4 | 0 | 0 | 0 |
| **Healthy-Leaf** | 0 | 9 | 0 | 0 | 0 |
| **Panama-Disease** | 0 | 0 | 5 | 0 | 0 |
| **Potassium-Deficiency** | 0 | 0 | 0 | 26 | 0 |
| **Yellow-Sigatoka** | 0 | 0 | 0 | 0 | 20 |

Figure: `reports/test_confusion_matrix.png`. Misclassified images: `reports/test_misclassified.csv` (4 of 82).

## How to read this

- The test set holds only 12 distinct photographs; a class with 1 or 2 groups has a result that rests on 1 or 2 photographs.
- Intervals are wide for the same reason, and a class with a single group has a meaningless interval.
- The cross-validation estimate, which uses all 100 train and validation photograph groups, is the better guide to expected performance on this dataset.
- Results describe this dataset only; other cameras, backgrounds and regions are untested.
- Confidence scores are not calibrated probabilities.
