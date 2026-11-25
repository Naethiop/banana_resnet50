# Stage 7 Findings: Final Model and Test Evaluation

## What was done
- Final model: frozen ResNet50 (ImageNet) plus a new softmax layer, trained on the train+validation pool (783 images, 100 photograph groups; the "pool" option), with augmented copies, balanced class weights and 15 epochs. Fine-tuning was not used (Stage 6b: not justified). The model package was saved before the test set was read; the test set was then evaluated once. The lock file `reports/test_evaluation_done.json` records this.
- Student's choice of training data: pool (recommended option). This departs from Chapter Three's early stopping on a validation set; the reason is that no validation data remains after pooling and 22 validation groups are too few for reliable early stopping.

## Results (test set: 82 images, 12 photograph groups)
- Accuracy 0.951; macro-F1 0.944 (95% group-bootstrap interval 0.899 to 1.000).
- Cross-validation estimate for the same method on the pool (Stage 6a, variant D): accuracy 0.720, macro-F1 0.696 (95% interval 0.595 to 0.844).
- Four of 82 images were misclassified, all Black Sigatoka predicted as Healthy leaf, all from one photograph.

## Interpretation (state with care)
- The test score is much higher than the cross-validation estimate. The test set is only 12 photographs: Black Sigatoka is one photograph of 21 images plus one image; Panama is one photograph; Potassium deficiency, Yellow Sigatoka two each; Healthy five. Per-class test results rest on one or two photographs.
- Post-hoc diagnostics (`python -m src.test_diagnostics`; they used test labels after the evaluation and changed nothing): 20 of 82 test images (24%) have a same-class train/val image at ResNet50-feature cosine similarity 0.80 or above, and 35 (43%) at 0.75 or above. These are Healthy 5 of 9, Panama 4 of 5, Yellow Sigatoka 11 of 20 at 0.80 and above, and none for Black Sigatoka or Potassium deficiency. None of the misclassified images is among them. So part of the test score likely reflects residual near-copies, but not all of it: Potassium deficiency (26 images) has no partner above 0.75 and was classified correctly.
- Error analysis: the four errors are cropped, close-up versions of one photograph of a mostly green Black Sigatoka leaf with faint small spots (confidence 0.42 to 0.68). The same photograph is classified correctly when the surrounding scene is visible (`reports/evidence/09_test_black_errors.png`). This is consistent with the risk noted in Stage 3 that close-up framing resembles the Healthy class, and with early-stage Black Sigatoka having subtle symptoms.
- The cross-validation figures (accuracy 0.72, macro-F1 0.70, 95% interval 0.60 to 0.84) are the better guide to expected performance on new photographs from this dataset's population. The test result should be reported as supplementary with these limits.

## What to write in Chapter Four
- Report both, say plainly that the test set contains 12 distinct photographs, give the photograph counts beside every per-class figure, and state that the test result was obtained once.
- Do not describe the system as "95% accurate".
- Disclose the post-hoc diagnostics and that the training pool included the validation images.
