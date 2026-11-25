# Stage 6a Notes: Group Cross-Validation and Uncertainty (head-only)

## Why this comes first
After regrouping, the data hold about 112 distinct photographs. The fixed validation set has 22 groups and the test set 12, so single-split numbers are too noisy to choose between options. Repeated group-stratified cross-validation (approved supplement) uses all 100 train and validation photograph groups for evidence, and group bootstrap intervals show how uncertain each number is.

## Design
- Pool: train and validation images only. The test set is excluded in code and never augmented, extracted or scored here.
- Folds: `StratifiedGroupKFold`, 4 folds (Panama has only 4 groups in the pool), 5 repeats with different seeds. Every photograph group is entirely inside or outside a fold's test part. Out-of-fold probabilities are averaged over repeats.
- Classifier: the same new softmax layer as the Stage 5 baseline on frozen ResNet50 features, fixed 15 epochs (no early stopping, since there is no spare validation data).
- Variants: A head only (reference); B balanced class weights; C augmented copies (4 per training image, features cached by `python -m src.features --augment 4`); D augmented copies plus balanced weights. Augmented copies belong to their source image's group, so they are only ever used for training, never for testing.
- Uncertainty: 95 percent intervals from 2000 resamples of whole photograph groups within each class (same resamples for every variant, so differences are paired).
- Selection rule, fixed before seeing results: A is kept unless another variant has higher macro-F1 and its paired difference interval from A lies entirely above zero. Otherwise the simpler variant stays.
- Reported per class: recall with interval, precision, number of photograph groups, and number of groups mostly classified correctly.

## Run
```
python -m pytest -q
python -m src.features --augment 4
python -m src.cv
```
Outputs: `data/features/resnet50_gap_aug.npz`, `reports/cv_report.md`, `reports/cv_results.json`.

## Limitations to state
- Cross-validation here covers the frozen-base head only. Fine-tuning the network inside cross-validation would take hours per fold on this laptop; it is handled separately in Stage 6b on the fixed split, with the caveat that its validation set is small.
- Scores describe this one dataset. They say nothing about other photographers, cameras or regions.
- Panama has only 4 photograph groups in the pool; holding one out leaves 3 for training. Its results are anecdotal.
- Some copies of a photograph may still be split across folds (see the residual similarity tables), which would make scores slightly optimistic.
