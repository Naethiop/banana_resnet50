"""Central project settings (Stage 2).

Settings only, no logic. Values marked INITIAL may be adjusted after
measurement, and every change is recorded in docs/decisions.md.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Model package files (written in Stage 7, read by the application)
MODEL_FILE = MODELS_DIR / "resnet50_banana.keras"
CLASS_NAMES_FILE = MODELS_DIR / "class_names.json"
MODEL_META_FILE = MODELS_DIR / "model_meta.json"

# Reproducibility
SEED = 42

# Approved in Chapter Three
IMAGE_SIZE = (224, 224)
SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}
FINE_TUNE_LEARNING_RATE = 1e-4   # "of the order of 1e-4" (Chapter Three)

# INITIAL training values, to be confirmed after the Stage 2 benchmark
BATCH_SIZE = 16
HEAD_LEARNING_RATE = 1e-3
MAX_EPOCHS_HEAD = 30
MAX_EPOCHS_FINE_TUNE = 30
EARLY_STOPPING_PATIENCE = 5

# Approved technical targets (decision 9)
MAX_UPLOAD_MB = 5
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
TARGET_RESPONSE_SECONDS = 5.0    # after the model has loaded

# Stage 4: group-aware split of the single approved dataset (Roboflow export only)
EXPORT_DIR = DATA_RAW / "Banana_Leaf_Disease_Detection_v2i_folder"
CLASS_MAP_FILE = DATA_RAW / "class_map.json"
AUDIT_CSV = REPORTS_DIR / "dataset_audit.csv"
SPLIT_CSV = DATA_SPLITS / "split_assignments.csv"
EXCLUDED_CSV = REPORTS_DIR / "excluded_images.csv"
SPLIT_REPORT = REPORTS_DIR / "split_report.md"
CLASS_NAMES_SPLIT_FILE = DATA_SPLITS / "class_names.json"
MIN_GROUPS_PER_SPLIT = 2          # each class needs at least this many photograph groups per split
SPLIT_SEARCH_TRIES = 5000         # seeded random assignments tried per class
CONFLICT_THRESHOLD = 0.90         # near-duplicates at or above this similarity must share a label (approved Stage 3 rule)
GROUP_THRESHOLD = 0.75            # same-label images at or above this similarity are one photograph group for splitting
LEAK_CHECK_THRESHOLDS = (0.75, 0.70, 0.65)   # residual-similarity check after the split

# Class names are written to data/splits/class_names.json by Stage 4
CLASS_NAMES = None

# Chosen from validation data by a documented method (Stage 8); never arbitrary
LOW_CONFIDENCE_THRESHOLD = None

# Stage 5: features, baseline and embedding-based duplicate check
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
FEATURES_FILE = FEATURES_DIR / "resnet50_gap.npz"
BASELINE_MODEL_FILE = MODELS_DIR / "baseline_model.keras"
BASELINE_META_FILE = MODELS_DIR / "baseline_meta.json"
BASELINE_REPORT = REPORTS_DIR / "baseline_val_report.md"
EMBEDDING_LEAK_REPORT = REPORTS_DIR / "embedding_leakage_report.md"
EVIDENCE_DIR = REPORTS_DIR / "evidence"
HEAD_DROPOUT = 0.3
HEAD_BATCH_SIZE = 32
AUGMENTATION = {"rotation_factor": 0.04, "brightness_factor": 0.2}   # +-14 degrees; +-20% brightness

# Stage 5b: groups also use embedding similarity (catches crops, zoom and rotations of one photograph)
EMBEDDING_LINK_THRESHOLD = 0.85   # same-class images with ResNet50-feature cosine >= this are linked into one group
EMBEDDING_LEAK_THRESHOLDS = (0.85, 0.80, 0.75)

# Stage 6a: group-stratified cross-validation on the train+val pool (the test set is never used)
CV_FOLDS = 4                     # Panama has only 4 photograph groups in the pool
CV_REPEATS = 5
CV_EPOCHS = 15                   # fixed budget: no early stopping inside CV (no spare validation data)
BOOTSTRAP_SAMPLES = 2000
AUG_COPIES = 4                   # augmented copies of each pool image whose features are cached
AUG_FEATURES_FILE = FEATURES_DIR / "resnet50_gap_aug.npz"
CV_REPORT = REPORTS_DIR / "cv_report.md"
CV_JSON = REPORTS_DIR / "cv_results.json"

# Stage 6b: fine-tuning of the last ResNet50 block, evaluated in one repeat of the group cross-validation
FT_EPOCHS = 6                    # fixed budget (no spare validation data inside cross-validation)
FT_UNFREEZE_PREFIX = "conv5_"    # layers of the last ResNet50 stage; batch-norm layers stay frozen
FT_REPORT = REPORTS_DIR / "finetune_cv_report.md"
FT_PARTIAL = REPORTS_DIR / "finetune_cv_partial.npz"

# Stage 7: final model package and the single evaluation on the test set
TEST_LOCK_FILE = REPORTS_DIR / "test_evaluation_done.json"
TEST_REPORT = REPORTS_DIR / "test_evaluation_report.md"
TEST_MISCLASSIFIED_CSV = REPORTS_DIR / "test_misclassified.csv"
TEST_CONFUSION_PNG = REPORTS_DIR / "test_confusion_matrix.png"

# Stage 8: prediction engine, low-confidence threshold and input (non-banana) check
CV_OOF_FILE = REPORTS_DIR / "cv_oof.npz"
THRESHOLD_REPORT = REPORTS_DIR / "threshold_report.md"
INPUT_CHECK_BANK_FILE = MODELS_DIR / "input_check_bank.npz"
NEGATIVES_DIR = DATA_RAW.parent / "negatives"
TARGET_ACCURACY = 0.85            # accepted predictions should reach this out-of-fold accuracy
MIN_COVERAGE = 0.50               # and at least this share of predictions must be accepted
INPUT_CHECK_BANANA_REJECT = 0.05  # the input check may reject at most this share of genuine banana images
INPUT_CHECK_MIN_AUROC = 0.95      # criteria for adopting an input check (fixed before seeing results)
INPUT_CHECK_MIN_REJECTION = 0.90

# Stage 9: Flask backend
MIN_IMAGE_SIDE = 64                  # pixels; smaller images are rejected as unusable
MAX_IMAGE_PIXELS = 50_000_000        # protects against decompression bombs

# Stage 11: optional features
import os as _os
ENABLE_HISTORY = _os.environ.get("BANANA_ENABLE_HISTORY", "true").lower() in {"1", "true", "yes", "on"}  # FR9; disable for public demo
HISTORY_DB = Path(_os.environ.get("BANANA_HISTORY_DB", str(PROJECT_ROOT / "app" / "instance" / "history.sqlite")))
CLIENT_MAX_SIDE = 1600                      # the browser shrinks larger photographs to this many pixels before upload
