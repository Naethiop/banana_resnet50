# Notes for Chapters Four and Five

Rule: record only what actually happened. Leave a section empty until real results exist. Never copy planned or expected values into this file.

## Environment actually used
(complete from reports/environment_report.txt)

## Dataset inspection findings (Stage 3)
(source, licence, counts, class distribution, duplicates, quality)

## Dataset arrangement approved (Stage 3 to 4)

## Preprocessing and split actually applied (Stage 4 to 5)

## Training configuration and results (Stages 5 to 6)
(training and validation results only)

## Final test evaluation (Stage 7)
(one evaluation on the untouched test set; per-class metrics; confusion matrix)

## Prediction engine, threshold selection and non-banana handling (Stage 8)

## Application implementation (Stages 9 to 11)

## System testing (Stage 12)

## Limitations to state
- Local-only deployment: farmers cannot use the system without access to the machine running it. Free hosting is possible future work only.
- Public dataset images may differ from real farm photographs.
- Confidence scores are not calibrated probabilities.
- Potassium deficiency is a nutrient disorder; the system classifies leaf appearance categories and does not diagnose causes.
- (add the non-banana limitation if it remains)
