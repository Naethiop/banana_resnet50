"""Pure helpers of Stage 8 (no TensorFlow needed)."""
import numpy as np

from src.cv import make_folds
from src.thresholds import (auroc, bootstrap_selective, choose_threshold, knn_oof_similarity, selective_point,
                            wilson)


def test_selective_point_and_choice():
    conf = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    correct = np.array([0, 0, 1, 0, 1, 1, 1, 1], dtype=bool)
    assert selective_point(conf, correct, 0.8) == (0.5, 1.0)
    t, how = choose_threshold(conf, correct, target=0.95, min_cov=0.5)
    assert 0.70 < t <= 0.80 and how == "target reached"      # smallest grid value that accepts only the top half
    t2, how2 = choose_threshold(conf, correct, target=1.01, min_cov=0.5)
    assert how2.startswith("target not reachable") and t2 <= 0.8


def test_auroc_and_wilson():
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.1], [0.9]) == 0.0
    assert abs(auroc([0.5, 0.5], [0.5, 0.5]) - 0.5) < 1e-9
    lo, hi = wilson(95, 95); assert lo > 0.95 and hi <= 1.0
    lo, hi = wilson(0, 10); assert lo == 0.0 or lo < 1e-9


def test_knn_oof_similarity_uses_training_side_only():
    X = np.array([[1, 0], [1, 0.01], [0, 1], [0.01, 1]], dtype=float)
    y = np.array([0, 0, 1, 1]); g = np.array([0, 1, 2, 3])
    sims = knn_oof_similarity(X, make_folds(y, g, 2, 0))
    assert sims.shape == (4,) and (sims <= 1.0 + 1e-9).all()


def test_bootstrap_selective_shapes():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 20); groups = np.repeat(np.arange(8), 5)
    conf = rng.uniform(0.4, 1, (3, 40)); correct = rng.random((3, 40)) < 0.7
    cov, acc = bootstrap_selective(y, groups, conf, correct, 0.7, 50, 1)
    assert cov.shape == (50,) and acc.shape == (50,) and ((cov >= 0) & (cov <= 1)).all()


def test_negative_source_groups_crops():
    from src.thresholds import negative_source
    assert negative_source("x/tile_21_png.rf.abc123.jpg") == "tile_21"
    assert negative_source("x/tile_21_png.rf.def456.jpg") == "tile_21"
    assert negative_source("x/IMG_001.jpg") == "IMG_001"
