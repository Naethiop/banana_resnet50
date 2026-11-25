# Deploy the banana leaf classifier on Render

This guide deploys the existing Flask application as a public demo. The app loads the trained Keras model when it starts. The model file is about 91 MiB and is excluded from Git unless explicitly allowed; `.gitignore` now allows only `models/resnet50_banana.keras` so it can be included in the deployment repository. Prediction history is disabled for the public service because the history page has no sign-in.

## 1. Check the deployment files

The repository now includes:

- `.python-version` to select Python 3.11 on Render.
- `requirements-deploy.txt` with the hosted app's pinned inference dependencies and Gunicorn.
- `wsgi.py` to create the Flask application for Gunicorn.
- `render.yaml` with the Render build/start commands, health check, 2 GB service size, and public-demo history setting.
- A `.gitignore` exception for the trained model required at startup.

The 2 GB plan is selected because this app loads TensorFlow and a ResNet50 model into memory. Render's free service has 512 MB RAM, spins down after 15 minutes without traffic, and loses local file changes on restart; it is not a dependable fit for this model or for persistent history. Check Render's current pricing before choosing a paid plan.

## 2. Create a GitHub repository

1. Sign in to GitHub as **Naethiop** and choose **New repository**.
2. Name it `banana_resnet50`.
3. Keep it **Private** if the model or project should not be public. Do not initialize it with a README, `.gitignore`, or licence; this project already has those files.
4. Create the repository and copy its HTTPS repository URL.

This project folder is not currently initialized as a Git repository, so run these commands in PowerShell from the folder containing `README.md`:

```powershell
git init
git add .
git status --short
```

Before continuing, confirm `models/resnet50_banana.keras` appears in the staged files, and that `.venv`, raw datasets, `models/baseline_model.keras`, and local SQLite files do not. Then commit and push (replace the URL with the one GitHub provided):

```powershell
git commit -m "Prepare banana classifier for deployment"
git branch -M main
git remote add origin https://github.com/Naethiop/banana_resnet50.git
git push -u origin main
```

The trained model is about 91 MiB. GitHub's regular Git uploads allow files below 100 MiB, so it fits currently; do not add the separate baseline model. If the trained model changes and grows beyond that limit, use Git LFS or external model storage instead of force-pushing a large file.

## 3. Create the Render service

1. Sign in to Render and connect your GitHub account if prompted.
2. In the Render dashboard, choose **New** → **Blueprint**.
3. Select the `banana-leaf-classifier` repository and the `main` branch.
4. Render will find `render.yaml`. Review the proposed service settings. The service is configured for the `1c-2g` plan (1 CPU, 2 GB RAM); review the displayed price before confirming.
5. Choose **Apply** to create the service and start its first deployment.

The Blueprint installs `requirements-deploy.txt`, launches `wsgi:app` with one Gunicorn worker, and checks `/health`. One worker avoids loading a second copy of the large model into memory. The model startup and first warm-up can take a few minutes.

## 4. Wait for the first deploy and open the site

1. Open the service's **Events** or **Logs** page in Render.
2. Wait for the build to finish and the deploy status to become **Live**. If it fails, inspect the first error in the build or runtime logs.
3. Click the `onrender.com` URL shown in the service overview.
4. Check that the home page loads and, optionally, visit `/health`; it should return `{"status":"ok"}`.
5. Try one JPG or PNG leaf image smaller than 5 MB and confirm a result appears.

## 5. Future updates

Edit files locally, then push the changes to the connected GitHub branch:

```powershell
git add .
git commit -m "Describe the update"
git push
```

Render automatically builds and deploys each push. Keep the model package together when replacing it: `models/resnet50_banana.keras`, `models/class_names.json`, `models/model_meta.json`, and `models/input_check_bank.npz` must match.

## Hosting notes

- Render free services sleep after 15 minutes idle, may take about a minute to wake, and lose local files whenever they restart or sleep. A persistent disk is available only on paid services. See [Render's free service limits](https://render.com/docs/free).
- The public deployment disables `/history`; the local app keeps its current history behavior. If history is needed online, add authentication and persistent storage before enabling it.
- Uploaded images are processed in memory by this app and are not saved as files.
- Anyone with the public URL can submit images. This demo is not configured for production traffic or agricultural diagnosis.
