"""Augmentation is defined for training only (Stage 5)."""
import numpy as np
import pytest


def test_augmentation_only_active_in_training():
    pytest.importorskip("keras")
    from src.train import build_augmentation
    aug = build_augmentation()
    x = np.random.default_rng(0).uniform(0, 255, (2, 224, 224, 3)).astype("float32")
    out_train = np.asarray(aug(x, training=True))
    out_eval = np.asarray(aug(x, training=False))
    assert out_train.shape == x.shape
    assert np.allclose(out_eval, x)                 # inference: unchanged
    assert not np.allclose(out_train, x)            # training: changed
