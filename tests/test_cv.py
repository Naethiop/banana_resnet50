"""Tests for the cross-validation helpers (Stage 6a); no TensorFlow needed."""
import numpy as np
from sklearn.metrics import f1_score

from src.cv import ci, group_bootstrap, group_correctness, make_folds, metrics, prefer_variant, run_cv


def synthetic(n_classes=3, groups_per_class=6, per_group=4, seed=0):
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    gid = 0
    for c in range(n_classes):
        centre = rng.normal(size=8) * 3
        for _ in range(groups_per_class):
            base = centre + rng.normal(size=8)
            for _ in range(per_group):
                X.append(base + rng.normal(size=8) * 0.1); y.append(c); g.append(gid)
            gid += 1
    return np.array(X), np.array(y), np.array(g)


def nearest_centroid(X, y, n):
    def fn(tr, te):
        cents = np.stack([X[tr][y[tr] == c].mean(axis=0) for c in range(n)])
        d = ((X[te][:, None, :] - cents[None]) ** 2).sum(axis=2)
        p = np.exp(-d); return p / p.sum(axis=1, keepdims=True)
    return fn


def test_folds_keep_groups_together_and_cover_every_item_once():
    X, y, g = synthetic()
    for tr, te in make_folds(y, g, 3, seed=1):
        assert set(g[tr]).isdisjoint(set(g[te]))
    seen = np.concatenate([te for _, te in make_folds(y, g, 3, seed=1)])
    assert sorted(seen) == list(range(len(y)))


def test_run_cv_returns_probabilities_for_every_item():
    X, y, g = synthetic()
    out = run_cv(len(y), y, g, nearest_centroid(X, y, 3), 3, 3, 2, 0)
    assert out.shape == (2, len(y), 3) and np.allclose(out.sum(axis=2), 1.0)
    assert (out.mean(axis=0).argmax(axis=1) == y).mean() > 0.9      # well-separated synthetic classes


def test_metrics_match_scikit_learn():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 4, 200); p = np.where(rng.random(200) < 0.7, y, rng.integers(0, 4, 200))
    m = metrics(y, p, 4)
    assert abs(m["macro_f1"] - f1_score(y, p, average="macro")) < 1e-9
    assert abs(m["accuracy"] - (y == p).mean()) < 1e-9


def test_group_bootstrap_shapes_determinism_and_paired_comparison():
    X, y, g = synthetic()
    perfect = y.copy(); noisy = np.where(np.arange(len(y)) % 2 == 0, y, (y + 1) % 3)
    b1 = group_bootstrap(y, g, {"perfect": perfect, "noisy": noisy}, 3, 200, seed=5)
    b2 = group_bootstrap(y, g, {"perfect": perfect, "noisy": noisy}, 3, 200, seed=5)
    assert b1["perfect"]["macro_f1"].shape == (200,) and b1["perfect"]["recall"].shape == (200, 3)
    assert np.array_equal(b1["noisy"]["macro_f1"], b2["noisy"]["macro_f1"])
    assert prefer_variant(b1, "noisy", "perfect") and not prefer_variant(b1, "perfect", "noisy")
    lo, hi = ci(b1["perfect"]["macro_f1"]); assert lo <= hi <= 1.0


def test_group_correctness_counts():
    y = np.array([0, 0, 0, 0, 1, 1]); g = np.array([0, 0, 1, 1, 2, 2]); p = np.array([0, 0, 1, 1, 1, 0])
    out = group_correctness(y, g, p, 2)
    assert out[0] == (2, 1) and out[1] == (1, 0)
