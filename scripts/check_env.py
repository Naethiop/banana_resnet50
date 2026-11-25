"""Environment check and optional speed benchmark (Stage 2).

Usage (from the project root, virtual environment active):
    python scripts/check_env.py
    python scripts/check_env.py --benchmark

Writes reports/environment_report.txt. The benchmark builds ResNet50 with
random weights (no download) and times it on THIS machine. Figures are
indicative measurements for planning, not project results.
"""
import argparse
import importlib
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINES = []


def say(text=""):
    print(text)
    LINES.append(text)


def version_of(module, dist):
    """Report the installed version; confirm the module really imports."""
    try:
        importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - report any import failure
        return f"NOT AVAILABLE ({type(exc).__name__}: {exc})"
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:  # noqa: BLE001
        return "installed (version unknown)"


def ram_gb():
    try:
        if os.name == "nt":
            import ctypes

            class Mem(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                            ("tpf", ctypes.c_ulonglong), ("apf", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                            ("ave", ctypes.c_ulonglong)]
            m = Mem()
            m.length = ctypes.sizeof(Mem)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.total / 1024 ** 3, m.avail / 1024 ** 3
        with open("/proc/meminfo") as fh:
            info = {k: int(v.split()[0]) for k, v in
                    (line.split(":") for line in fh if ":" in line)}
        return info["MemTotal"] / 1024 ** 2, info.get("MemAvailable", 0) / 1024 ** 2
    except Exception:  # noqa: BLE001
        return None


def benchmark(batch_size, steps):
    import numpy as np
    import tensorflow as tf

    say("\n--- Benchmark (random weights, 224x224x3, 5 classes) ---")
    model = tf.keras.applications.ResNet50(weights=None, input_shape=(224, 224, 3), classes=5)

    x1 = np.random.rand(1, 224, 224, 3).astype("float32") * 255.0
    model(x1, training=False)  # warm-up
    runs = 10
    start = time.perf_counter()
    for _ in range(runs):
        model(x1, training=False)
    infer = (time.perf_counter() - start) / runs
    say(f"Inference, 1 image (model already built): {infer:.3f} s per image")

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    xb = np.random.rand(batch_size, 224, 224, 3).astype("float32") * 255.0
    yb = np.random.randint(0, 5, size=(batch_size,))
    model.train_on_batch(xb, yb)  # warm-up
    start = time.perf_counter()
    for _ in range(steps):
        model.train_on_batch(xb, yb)
    step = (time.perf_counter() - start) / steps
    say(f"Training step, batch {batch_size}, ALL layers trainable: {step:.2f} s per step "
        f"({step / batch_size:.2f} s per image)")
    say(f"Indicative time for one pass over 657 images (all layers trainable): "
        f"{657 / batch_size * step / 60:.1f} minutes")
    say("Fine-tuning only the upper blocks is faster than this worst case. "
        "These figures guide the Stage 6 decision on local training versus Colab.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    say("=== Environment report ===")
    say(f"Python: {sys.version.split()[0]} ({platform.architecture()[0]})")
    say(f"Platform: {platform.platform()}")
    say(f"Processor: {platform.processor() or 'unknown'}")
    say(f"CPU threads: {os.cpu_count()}")
    ram = ram_gb()
    say(f"RAM total/available (GB): {ram[0]:.1f} / {ram[1]:.1f}" if ram else "RAM: unknown")
    say(f"In virtual environment: {sys.prefix != getattr(sys, 'base_prefix', sys.prefix)}")
    say("\n--- Packages ---")
    for module, dist in [("tensorflow", "tensorflow"), ("keras", "keras"), ("numpy", "numpy"),
                         ("pandas", "pandas"), ("sklearn", "scikit-learn"),
                         ("matplotlib", "matplotlib"), ("PIL", "pillow"),
                         ("flask", "flask"), ("pytest", "pytest")]:
        say(f"{dist}: {version_of(module, dist)}")

    try:
        import tensorflow as tf
        say(f"\nGPUs visible to TensorFlow: {tf.config.list_physical_devices('GPU') or 'none (CPU only)'}")
        say(f"Quick TensorFlow test: {float(tf.reduce_sum(tf.random.normal([1000, 1000]))) != 0.0}")
    except Exception as exc:  # noqa: BLE001
        say(f"\nTensorFlow could not be used: {exc}")
        if args.benchmark:
            say("Benchmark skipped.")
            args.benchmark = False

    if args.benchmark:
        benchmark(args.batch_size, args.steps)

    out = ROOT / "reports" / "environment_report.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    say(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
