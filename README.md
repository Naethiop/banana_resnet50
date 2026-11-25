# AI-Based Banana Crop Disease Detection System Using the ResNet50 Deep Learning Model

Undergraduate project. This system supports preliminary identification of banana leaf conditions. It does not replace laboratory diagnosis or qualified agricultural advice.

Current stage: Stage 2 (environment and repository). See `docs/decisions.md`.

## Deploy a live demo

Render deployment files and the step-by-step setup are in [`DEPLOYMENT.md`](DEPLOYMENT.md). The hosted setup requires a GitHub repository containing the trained model package; see the guide before pushing, as the model is about 91 MiB.

## Setup on Windows (PowerShell)

Run from the project folder (the folder containing this README).

```
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_env.py
python scripts/check_env.py --benchmark
pip freeze > requirements-lock.txt
python -m pytest -q
```

- If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.
- TensorFlow on Windows needs the Microsoft Visual C++ Redistributable (2015 to 2022). Install it first if TensorFlow fails to import.
- On Linux or macOS use `python3.11 -m venv .venv` and `source .venv/bin/activate`.

## What to expect

- `check_env.py` prints your Python, RAM, CPU, package versions and whether TensorFlow imports, and saves `reports/environment_report.txt`. Expect "GPUs visible to TensorFlow: none (CPU only)" on this laptop.
- `--benchmark` takes a few minutes and times ResNet50 on your machine using random weights (no download). The figures are measurements for planning only.
- `pytest` should report 3 passed.
- `requirements-lock.txt` records the exact installed versions. Keep it; the same TensorFlow version must be used to load the trained model.

## Git (optional but recommended)

```
git init
git add .
git commit -m "Stage 2: project scaffold"
```

## Folder guide

See `docs/decisions.md` for the approved plan. Data, models and uploads are excluded from Git by `.gitignore`.
