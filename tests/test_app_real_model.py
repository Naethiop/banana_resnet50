"""End-to-end check with the real model package (skipped when TensorFlow or the model is missing)."""
import pytest

from src import config


def test_real_model_through_the_web_backend():
    pytest.importorskip("keras")
    if not config.MODEL_FILE.exists():
        pytest.skip("model package not found")
    from app.app import create_app
    from src.features import load_split_table

    tab = load_split_table()
    path = tab[tab.assigned_split == "val"].path.iloc[0]          # a validation image, never a test image
    client = create_app().test_client()
    with open(path, "rb") as fh:
        r = client.post("/predict", data={"image": (fh, path.name)}, content_type="multipart/form-data")
    text = r.get_data(as_text=True)
    assert r.status_code == 200 and "Model confidence" in text and "does not replace laboratory" in text
