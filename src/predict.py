"""Stage 8: the prediction engine. Loads the frozen model package once and classifies one image at a time.

Independent of Flask. Usage in code:
    predictor = Predictor()
    result = predictor.predict("leaf.jpg")        # a path or a file-like object

    python -m src.predict --benchmark             # timing on this machine (uses validation images, never test images)

Result keys: label, confidence, top (best three), low_confidence (None if no threshold is set), unsuitable
(None if no input check is adopted), seconds. The confidence is the model's largest softmax output. It is not
a calibrated probability and a high value does not prove the label is right.
"""
import argparse
import time

import numpy as np

from src import config
from src.model_io import load_package
from src.preprocessing import load_rgb, preprocess_batch


class Predictor:
    def __init__(self):
        self.model, self.class_names, self.meta = load_package()
        n = len(self.class_names)
        shapes = {tuple(l.output.shape): l for l in self.model.layers if hasattr(l, "output") and l.output is not None}
        self.base, self.head = shapes.get((None, 2048)), shapes.get((None, n))
        if self.base is None or self.head is None:
            raise RuntimeError("Unexpected model structure; cannot find the feature extractor and the head.")
        self.low_conf = self.meta.get("low_confidence_threshold")
        self.input_check = self.meta.get("input_check")
        self.bank = None
        if self.input_check:
            self.bank = np.load(config.INPUT_CHECK_BANK_FILE)["bank"]
        self._self_check()

    def _forward(self, x):
        feats = self.base(x, training=False)
        probs = self.head(feats, training=False)
        return np.asarray(feats)[0], np.asarray(probs)[0]

    def _self_check(self):
        """The split base+head path must equal the saved full model; also warms the model up."""
        x = preprocess_batch(np.random.default_rng(0).uniform(0, 255, (1, 224, 224, 3)).astype("float32"))
        _, p = self._forward(x)
        full = np.asarray(self.model(x, training=False))[0]
        if float(np.abs(p - full).max()) > 1e-4:
            raise RuntimeError("Prediction path disagrees with the saved model.")

    def features_and_probs(self, source):
        return self._forward(preprocess_batch(load_rgb(source)[None]))

    def predict(self, source):
        t0 = time.perf_counter()
        feats, probs = self.features_and_probs(source)
        order = np.argsort(-probs)
        top = [(self.class_names[i], float(probs[i])) for i in order[:3]]
        conf = float(probs[order[0]])
        unsuitable = None
        if self.bank is not None:
            f = feats / (np.linalg.norm(feats) + 1e-12)
            unsuitable = bool(float((self.bank @ f).max()) < self.input_check["threshold"])
        return {"label": top[0][0], "confidence": conf, "top": top,
                "low_confidence": None if self.low_conf is None else conf < self.low_conf,
                "unsuitable": unsuitable, "seconds": time.perf_counter() - t0}


def benchmark():
    from src.features import load_split_table
    t0 = time.perf_counter()
    pr = Predictor()
    load_s = time.perf_counter() - t0
    tab = load_split_table()
    paths = list(tab[tab.assigned_split == "val"].path[:20])          # validation images only
    times = []
    first = pr.predict(paths[0])["seconds"]
    for p in paths:
        times.append(pr.predict(p)["seconds"])
    print(f"Model package load and self-check: {load_s:.1f} s")
    print(f"First prediction: {first:.2f} s; {len(times)} predictions: mean {np.mean(times):.2f} s, "
          f"95th percentile {np.percentile(times, 95):.2f} s, maximum {np.max(times):.2f} s "
          f"(target: about {config.TARGET_RESPONSE_SECONDS:.0f} s after the model has loaded)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--status", action="store_true", help="show the adopted thresholds (does not load the model)")
    args = ap.parse_args()
    if args.status:
        import json
        meta = json.loads(config.MODEL_META_FILE.read_text(encoding="utf-8"))
        ic = meta.get("input_check")
        print(f"Low-confidence threshold: {meta.get('low_confidence_threshold')} "
              f"({'adopted' if meta.get('low_confidence_threshold') is not None else 'NOT adopted: no low-confidence warning will appear'})")
        print("Input check: " + (f"adopted (similarity threshold {ic['threshold']:.3f})" if ic else
                                 "NOT adopted: no unsuitable-image warning will appear"))
        print(f"Model file hash recorded: {str(meta.get('model_sha256'))[:16]}...")
    elif args.benchmark:
        benchmark()
    else:
        print(__doc__)
