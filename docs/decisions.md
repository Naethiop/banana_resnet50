# Project Decisions Log (Stage 1: Decision Freeze)

Project: AI-Based Banana Crop Disease Detection System Using the ResNet50 Deep Learning Model
Status: plan confirmed by the student. Changes after this point are recorded in the change log below, with a reason and an approval.

## 1. Approved decisions

| # | Topic | Decision |
|---|---|---|
| 1 | Disease versus nutrient disorder | Title unchanged. The interface says "banana leaf condition" or "classification category". Potassium deficiency is explained as a nutrient disorder, not an infectious disease |
| 2 | Input scope | Banana leaf images only, for design, training and evaluation |
| 3 | Database | None by default. FR9 (history) is desirable; if implemented, SQLite text records only |
| 4 | Stakeholders | Farmers and agricultural extension workers; no administrator |
| 5 | Research gap | Narrowed to an accessible, locally deployable classification application on the selected dataset; no "never evaluated" claim |
| 6 | Privacy | Uploaded images are never stored permanently; temporary files are deleted after prediction |
| 7 | Preprocessing | ResNet50-specific preprocessing for the ImageNet weights, identical in training, validation, test and application |
| 8 | Stages | Twelve stages followed in order; a stage is implemented, tested and reviewed before the next |
| 9 | Targets | 5 MB maximum; JPG, JPEG, PNG; one leaf image per prediction; about 5 seconds after model load; confidence is not proof of correctness and is not called calibrated; low-confidence threshold chosen from validation data by a documented method |
| 10 | Non-banana images | Investigated in Stage 8 only if suitable negative images exist; otherwise a stated limitation and interface warning. FR7 stays desirable |
| 11 | Dataset | Candidate only until the Stage 3 inspection report is approved |
| 12 | Deployment | Local Flask application; limitation documented in Chapter Four |
| 13 | Environment | Modest laptop; Colab optional and not assumed |

## 2. Development environment (reported by the student)

- Laptop: Dell Latitude E7240, Intel Core i5 (exact model to be confirmed by `scripts/check_env.py`)
- RAM: 8 GB; free disk space: 45 GB; 64-bit operating system
- Python: 3.11.0
- Google account available (Colab optional)
- GPU: none assumed. `scripts/check_env.py` reports what TensorFlow actually detects.

Planning consequences (to be confirmed by measurement, not assumed):
- Local machine: dataset audit and preparation, application development, inference and the Flask demonstration.
- Training: head training may be feasible locally; the Stage 6 fine-tuning location is decided after the benchmark. Colab is the preferred option if it is available, with a reduced local run as the fallback.
- Close other programs during training; the 8 GB RAM limit may require a smaller batch size.

## 3. Candidate dataset (not confirmed)

- Cited by Alanazi (2025) as: "Banana Leaf Disease Detection Dataset", Roboflow Universe, https://universe.roboflow.com/rkn/banana-leaf-disease-detection-twfnp
- The paper reports 938 images in five classes. These figures are from the paper and are NOT verified.
- Licence: NOT verified. The page could not be opened during planning. Record the licence shown on the page before downloading.
- A separate Roboflow project titled "Banana Disease Detection" (author "Thesis") lists the same five class names. No relationship to the project above has been established; it is not to be used unless compared and approved.
- Every item in the Stage 3 inspection report is to be completed from the real files.

## 4. Change log

| Date | Change | Reason | Approved by |
|---|---|---|---|
| | | | |
