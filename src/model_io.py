"""Save and load the model package: the Keras model, the ordered class names and the metadata.

The application depends only on this package, so a retrained model can replace it without changing the
interface (maintainability requirement).
"""
import hashlib
import json
import platform
from datetime import date

from src import config


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_package(model, class_names, meta):
    import keras
    config.MODELS_DIR.mkdir(exist_ok=True)
    model.save(str(config.MODEL_FILE))
    config.CLASS_NAMES_FILE.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    meta = dict(meta)
    meta.update({"date": str(date.today()), "python": platform.python_version(), "keras": keras.__version__,
                 "class_names": class_names, "model_sha256": sha256_of(config.MODEL_FILE)})
    try:
        import tensorflow as tf
        meta["tensorflow"] = tf.__version__
    except Exception:  # noqa: BLE001
        pass
    config.MODEL_META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_package():
    """Returns (model, class_names, meta)."""
    import keras
    model = keras.saving.load_model(str(config.MODEL_FILE))
    names = json.loads(config.CLASS_NAMES_FILE.read_text(encoding="utf-8"))
    meta = json.loads(config.MODEL_META_FILE.read_text(encoding="utf-8"))
    return model, names, meta
