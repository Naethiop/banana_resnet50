# Stage 6b Notes: Fine-Tuning Compared with the Head-Only Model

## Decisions taken from Stage 6a (cross-validation of the head-only model)
- Augmentation helps: macro-F1 0.688 versus 0.634 (paired difference +0.052, 95% interval +0.010 to +0.094).
- Balanced class weights alone make no detectable difference (+0.014, interval -0.010 to +0.034). Variant D (augmentation plus weights, 0.696) was chosen by the pre-set rule; the extra benefit of weights over C is untested.
- Head-only performance with whole photographs held out is about macro-F1 0.70 (95% interval 0.60 to 0.84) and accuracy 0.72. Panama (4 photograph groups) is anecdotal.

## Stage 6b design
- Chapter Three's two stages: stage 1 trains the new layer on the frozen base; stage 2 unfreezes the upper blocks and fine-tunes at a low learning rate (1e-4). Here the last ResNet50 stage (`conv5_*` layers) is unfrozen; batch-norm layers stay frozen.
- Evaluated inside one repeat of the same 4-fold group cross-validation, so each fold's fine-tuned and head-only predictions are paired, and the test fold plays no part in training or in choosing epochs.
- Fixed 6 epochs (no early stopping, no spare validation data). The count is a stated setting, not tuned.
- Rule fixed in advance: fine-tuning is "justified" only if the paired macro-F1 difference interval lies entirely above zero; otherwise the head-only model is kept ("fine-tune only when justified", project brief).

## Run
```
python -m pytest -q
python -m src.finetune --time-check
python -m src.finetune
```
Send the time-check output first. The full run is expected to take about an hour or more; keep the laptop plugged in and awake. If it stops, `python -m src.finetune --resume` continues from the last finished fold.

## Limitations
- One repeat only; fine-tuning is far too slow on this laptop to repeat five times.
- Untested until run: the assistant could not execute TensorFlow code.
