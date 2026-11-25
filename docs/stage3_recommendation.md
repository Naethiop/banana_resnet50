# Stage 3 Findings and Recommended Dataset Arrangement (for approval)

Basis: the downloaded export `Banana_Leaf_Disease_Detection_v2i_folder.zip`, audited with `scripts/audit_dataset.py`. Full measurements are in `reports/dataset_inspection_report.md` and `reports/dataset_audit.csv`. Pictures that support the findings are in `reports/evidence/`.

Status: nothing below is applied. Stage 4 does not start until you approve an arrangement.

## 1. What is now verified, and what is not

| Item | Status |
|---|---|
| 938 images, all readable JPEG, RGB, 640 x 640 pixels | Verified from the files |
| Supplied split 657 train, 188 validation, 93 test | Verified; matches the counts reported by Alanazi (2025), so that paper probably used this split (not confirmed) |
| Five classes; class sizes 301, 241, 220, 91, 85 | Verified (class names inferred, see 2.1) |
| Exact duplicate files | None |
| Licence CC BY 4.0 | Reported by you from the dataset page; not independently verified |
| Export settings | README says only auto-orientation was applied and no augmentation |

CC BY 4.0 requires attribution. Cite the dataset in the project report with the creator and title exactly as the Roboflow page shows them, and note the export version (v2).

## 2. Findings that affect the plan

### 2.1 Class folders have unusable names
The export uses folders named `1`, `2`, `3`, `4` and `1 0 0 0 1 1 1 1 0 0 0`. I mapped them from the original file-name prefixes (almost every file agrees with its folder) and checked sample images by eye: 1 = Healthy-Leaf, 2 = Panama-Disease, 3 = Potassium-Deficiency, 4 = Yellow-Sigatoka, the long name = Black-Sigatoka. This is an inference. You must confirm it against the class names shown on the Roboflow page (`data/raw/class_map.json`).

### 2.2 The images are mostly repeated photographs (the most important finding)
Many images are the same photograph with different brightness, mirror flips, 90-degree rotations or crops, so the test set is not independent of the training set (see `reports/evidence/03` and `04`).
- The 938 images form about **230 distinct photographs** (a minimum; the method can miss heavily cropped copies).
- Of the 129 groups that contain several images, **105 are spread across more than one split**.
- **86 of the 93 test images** and 168 of the 188 validation images have a near-duplicate in another split.
- Distinct photographs per class: Healthy 91, Potassium deficiency 45, Yellow Sigatoka 40, Black Sigatoka 38, Panama 24.

Using the supplied split would make test results mostly measure recognition of photographs already seen in training. It would not support a claim of generalisation.

### 2.3 Some photographs carry two different class labels
Eight near-duplicate groups (72 images) contain the same photograph under different labels, mainly Black Sigatoka and Potassium deficiency (`reports/evidence/04`). Separately, four images have a file name that disagrees with the folder (`reports/evidence/02`). My visual impression, which is not a diagnosis, is that two of the four look wrong in their folders (a lesioned leaf in Healthy; a uniformly green leaf in Potassium deficiency). In total 73 images in 9 groups are flagged. I cannot decide which label is biologically right.

### 2.4 The Panama category is not only leaf blades
In the contact sheet of all 85 Panama images (`reports/evidence/05`), by eye roughly 40 to 50 percent show pseudostem, leaf-sheath or petiole views, not a leaf blade. This is an estimate, not a count. It conflicts with decision 2 (leaf images only).

### 2.5 Possible shortcut for the Healthy class
In the contact sheet of all 91 Healthy images (`reports/evidence/06`), nearly all are close-ups of a leaf blade filling the frame with little background. The disease classes more often show whole leaves with field backgrounds. A model could partly learn framing and background instead of disease. This cannot be removed, only reported as a limitation and watched for in the confusion matrix.

### 2.6 What the audit cannot tell us
It does not judge whether any label is biologically correct, and the duplicate detector is a heuristic (see Section 8 of the report). The cause of the repeated photographs is unknown; the images look like augmented copies, but the export README says none were applied, so they were probably in the uploaded images already.

## 3. Recommended arrangement

1. **Confirm the class mapping** from the Roboflow page (2.1).
2. **Do not use the supplied split.** Pool all images and re-split them in Stage 4.
3. **Exclude the 73 flagged images** (9 groups) from training, validation and testing, and list them in `reports/excluded_images.csv`. If a qualified person later adjudicates them they can be restored. This leaves 865 images in 221 groups: Black Sigatoka 211 (30 groups), Healthy 90 (90), Panama 85 (24), Potassium deficiency 259 (37), Yellow Sigatoka 220 (40).
4. **Split by photograph group, stratified by class**, with a fixed seed. Every image of a group goes into one partition. The 70/20/10 target is applied to groups per class. Expect only about 2 to 9 distinct photographs per class in the test set. Image counts in each partition will differ from 70/20/10.
5. **Report image counts and distinct-photograph counts** beside every metric, and state the uncertainty. This is a change of reporting detail, not of the approved metrics.
6. **Panama scope (your decision):**
   - P1 (recommended): keep all Panama images; amend the scope wording to say the Panama category includes leaf-sheath and pseudostem views present in the dataset; state it as a limitation.
   - P2: remove non-leaf-blade Panama images. This probably leaves fewer than about 12 photographs, too few for a separate class.
7. **Class weights and augmentation** are decided in Stage 5 using distinct-photograph counts, not raw image counts, because the raw counts overstate how much independent data each class has.

### Optional supplements (need your explicit approval)
- Uncertainty reporting for per-class metrics (for example bootstrap by group).
- Repeated group-stratified cross-validation with the head-only model, to give more stable estimates than a single tiny test set.

## 4. Alternatives

- **A. Use the supplied split unchanged.** Not recommended (2.2).
- **B. Add independent images** (your own photographs or another dataset). It would enlarge the number of distinct photographs, but it needs approval, labelling expertise and licence checks, and costs time. A second dataset would not necessarily contain the same classes.

## 5. Effect on the approved chapters (wording for your approval; nothing changed yet)

- **Chapter Three, 3.2.3 (data splitting):** keep 70/20/10 stratified, and add that near-duplicate images are grouped so that each photograph lies in one partition. Remove the sentence that justifies the proportions by citing the supplied split of the same dataset only if you choose to.
- **Chapter Three, scope (Panama):** depends on decision P1 or P2.
- **Chapter Two or Four:** the published results of Alanazi (2025) probably used the supplied split, which our audit shows to contain near-duplicates across partitions. A cautious sentence is suggested: published results on that split may overstate performance on new photographs. Wording to be agreed with your supervisor, because it touches another author's work.
- **Chapter Four:** record the audit, exclusions and limitations (2.3 to 2.6).

## 6. Decisions needed from you

1. Is the class mapping in 2.1 correct (compare with the Roboflow page)?
2. Approve the group-aware re-split instead of the supplied split?
3. Exclude the 73 flagged images, or keep any for adjudication?
4. Panama: P1 or P2?
5. Are the optional supplements wanted?
6. Do you want to look for extra independent images (Alternative B), or work with this dataset only?

## 7. To add to docs/decisions.md after your approval

- Dataset: Banana Leaf Disease Detection v2, Roboflow, CC BY 4.0 (reported), 938 images, five classes, mapping as in `class_map.json`.
- Supplied split rejected because of near-duplicate leakage; group-aware re-split adopted.
- Exclusions and Panama decision as approved.
