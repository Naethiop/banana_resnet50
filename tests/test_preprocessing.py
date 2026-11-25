"""Tests for the shared preprocessing (Stage 5)."""
import numpy as np
import pytest
from PIL import Image

from src.preprocessing import IMAGENET_BGR_MEAN, load_rgb, preprocess_batch


def _save(path, arr, mode="RGB"):
    Image.fromarray(arr.astype("uint8"), mode).save(path)
    return path


def test_load_rgb_shape_dtype_range(tmp_path):
    rng = np.random.default_rng(0)
    p = _save(tmp_path / "a.png", rng.integers(0, 256, (200, 300, 4)), "RGBA")   # non-square, with alpha
    x = load_rgb(p)
    assert x.shape == (224, 224, 3) and x.dtype == np.float32
    assert x.min() >= 0 and x.max() <= 255


def test_centre_crop_keeps_proportions(tmp_path):
    arr = np.zeros((200, 400, 3))
    arr[:, :200] = (255, 0, 0)       # left half red
    arr[:, 200:] = (0, 0, 255)       # right half blue
    x = load_rgb(_save(tmp_path / "b.png", arr))
    assert x[112, 20, 0] > 200 and x[112, 20, 2] < 50        # left side red
    assert x[112, 200, 2] > 200 and x[112, 200, 0] < 50      # right side blue


def test_preprocess_is_bgr_mean_subtraction():
    red = np.zeros((1, 224, 224, 3), dtype=np.float32)
    red[..., 0] = 255                                         # RGB red
    out = preprocess_batch(red)
    assert np.allclose(out[0, 0, 0], [0 - 103.939, 0 - 116.779, 255 - 123.68], atol=1e-3)
    assert np.allclose(red[..., 0], 255)                      # input untouched


def test_same_file_same_output(tmp_path):
    rng = np.random.default_rng(1)
    p = _save(tmp_path / "c.png", rng.integers(0, 256, (224, 224, 3)))
    assert np.array_equal(preprocess_batch(load_rgb(p)[None]), preprocess_batch(load_rgb(p)[None]))


def test_matches_keras_resnet50_preprocessing():
    keras = pytest.importorskip("keras")
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 255, (2, 224, 224, 3)).astype("float32")
    ref = np.asarray(keras.applications.resnet50.preprocess_input(x.copy()))
    assert np.allclose(preprocess_batch(x), ref, atol=1e-3)
