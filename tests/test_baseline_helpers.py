"""Tests for the pure helpers of Stage 5 (no TensorFlow needed)."""
import numpy as np

from src.leakage_embeddings import cosine_matrix, cross_split_pairs, reference_values
from src.train import choose_setting, class_weight_dict, evaluate_predictions


def test_class_weights_favour_rare_classes():
    y = np.array([0] * 8 + [1] * 2)
    w = class_weight_dict(y, 2)
    assert w[1] > w[0] and abs(w[0] * 8 + w[1] * 2 - 10) < 1e-9


def test_evaluate_predictions_counts_and_groups():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    groups = np.array([5, 5, 6, 7, 8, 9])
    r = evaluate_predictions(y_true, y_pred, ["a", "b", "c"], groups)
    assert abs(r["accuracy"] - 4 / 6) < 1e-9
    assert r["per_class"]["a"]["images"] == 2 and r["per_class"]["a"]["photograph_groups"] == 1
    assert r["per_class"]["b"]["photograph_groups"] == 2
    assert np.array(r["confusion"]).sum() == 6


def test_choose_setting_prefers_none_on_tie():
    assert choose_setting({"none": {"macro_f1": 0.8}, "balanced": {"macro_f1": 0.8}}) == "none"
    assert choose_setting({"none": {"macro_f1": 0.7}, "balanced": {"macro_f1": 0.8}}) == "balanced"


def test_cross_split_pairs_same_class_only():
    f = np.array([[1, 0], [1, 0.01], [1, 0.02], [0, 1]], dtype=float)
    sim = cosine_matrix(f)
    labels = np.array(["a", "a", "b", "a"]); splits = np.array(["train", "test", "val", "train"])
    pairs = cross_split_pairs(sim, labels, splits, 0.99)
    assert pairs == [(0, 1)]


def test_reference_values_shapes():
    f = np.array([[1, 0], [1, 0.05], [0, 1], [0.05, 1]], dtype=float)
    sim = cosine_matrix(f)
    within, diff = reference_values(sim, np.array([0, 0, 1, 1]), np.array(["a", "a", "b", "b"]))
    assert len(within) == 2 and len(diff) == 4 and within.min() > 0.99 and diff.max() < 0.1
