# Test Evaluation Diagnostics (descriptive, after the single evaluation)

These diagnostics used the test labels after the evaluation. They did not change the model.

## Test images with a near-copy in the train and validation pool

| Cosine threshold | Test images with a same-class pool image at or above it | of which misclassified |
|---|---|---|
| 0.85 | 0 of 82 | 0 |
| 0.8 | 20 of 82 | 0 |
| 0.75 | 35 of 82 | 0 |

## Per class

| Class | Test images | Photograph groups | Lowest | Median | Highest | Images at or above 0.80 | Misclassified |
|---|---|---|---|---|---|---|---|
| Black-Sigatoka | 22 | 2 | 0.39 | 0.72 | 0.75 | 0 | 4 |
| Healthy-Leaf | 9 | 5 | 0.71 | 0.81 | 0.85 | 5 | 0 |
| Panama-Disease | 5 | 1 | 0.78 | 0.83 | 0.84 | 4 | 0 |
| Potassium-Deficiency | 26 | 2 | 0.57 | 0.67 | 0.75 | 0 | 0 |
| Yellow-Sigatoka | 20 | 2 | 0.76 | 0.81 | 0.85 | 11 | 0 |

## Misclassified images by photograph group

| Class | Group | Images in group (test) | Misclassified |
|---|---|---|---|
| Black-Sigatoka | 83 | 21 | 4 |

## Reading this

- 'Highest similarity to a same-class pool image' of 0.80 and above, in earlier pair inspections, usually meant a crop, zoom or recolouring of the same scene. Test images with such partners may be easier than truly new photographs.
- Classes whose test images have no such partners are a fairer check.
