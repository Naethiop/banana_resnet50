# Stage 7 Notes: Final Model and Single Test Evaluation

## Model chosen (decisions from Stage 6)
- Frozen ResNet50 (ImageNet) with a new softmax layer; training with augmented copies and balanced class weights (variant D, cross-validation macro-F1 0.696, 95% interval 0.595 to 0.844).
- Fine-tuning not used: Stage 6b gave a paired macro-F1 difference of -0.038 (95% interval -0.081 to +0.011), so the pre-set rule did not justify it. Training loss fell to near zero without better held-out results (the network memorised the ~100 training photographs).
- Class weights alone made no detectable difference in cross-validation (+0.014, interval -0.010 to +0.034). They are kept because they cost nothing, not because they were shown to help.

## Decision needed before running: what data trains the final model?
- **Option pool (recommended):** train on all train and validation images (783 images, 100 photograph groups), with the epoch count fixed by cross-validation. It gives 4 Panama groups instead of 3 and more Potassium and Yellow groups. The test set stays untouched. It differs from Chapter Three's plan of early stopping on a validation set: there is no validation data left after pooling, and early stopping on 22 groups would be unreliable.
- **Option train:** train on the train split only (78 groups), as originally planned. Cleaner in following the approved text, but less data.
- Either way the test set is not used for training, model selection or threshold choice.

## Safeguards
- The model package (`models/resnet50_banana.keras`, `class_names.json`, `model_meta.json`) is saved before the test set is read.
- The script checks that the evaluation path (full model on test images) agrees with the cached-feature path.
- A lock file (`reports/test_evaluation_done.json`) records the single evaluation. A second run is refused unless `--allow-rerun` is passed, and any rerun must be disclosed.

## Run (after the decision above)
```
python -m pytest -q
python -m src.evaluate               # option pool
python -m src.evaluate --train-on train   # option train
```

## Limits to state
- The test set contains only 12 photograph groups; Panama has one. Per-class test results are anecdotal and the cross-validation figures are the better guide.
- Scores describe this dataset only. Other cameras, backgrounds and regions are untested.
- Confidence scores are not calibrated probabilities.
