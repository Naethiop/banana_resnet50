# Stage 9 Notes: Flask Backend

## Decisions applied
- Low-confidence threshold 0.94 (rule in Stage 8; target accuracy not reachable, so the largest threshold that keeps half of the predictions), shown with softer wording that does not imply the result is wrong (student decision).
- Input check: similarity to training images, adopted only as a soft warning if the student approves (see below). Evidence: all 95 aerial tiles rejected; of 23 ordinary photographs, 16 rejected (70%, 95% interval 0.49 to 0.84) and the 7 missed were leaves or plants (cassava, mango, papaya, Dracaena, a tomato-type leaf, one unnamed PNG), mostly labelled Yellow Sigatoka at 0.98 or higher. It catches obviously non-leaf scenes but not leaves of other plants.
- A permanent scope note appears on every page: other plants will still receive one of the five labels.

## What was built (tested here with a fake predictor; the real model is tested on your laptop)
- `app/app.py`: `create_app(predictor=None)`; routes `/` (home and upload form), `/predict` (POST), `/health`; handlers for 404, 405, 413 and unexpected errors (no internal details shown).
- `app/validators.py`: file present, extension JPG/JPEG/PNG, size at most 5 MB (read in memory, never stored), real file type from the first bytes, image opens without error, minimum 64 pixels per side, one image per request.
- `app/results.py`: category names and explanations (`app/content/conditions.json`, based on the descriptions in Chapter Two), messages, confidence note, and a small preview made in memory.
- Privacy: uploads are processed in memory and never written to disk, so nothing needs deleting; the preview is embedded in the page. Nothing about the image is logged.
- Headers: `X-Content-Type-Options: nosniff`, `Cache-Control: no-store`.
- Minimal templates and CSS only; Stage 10 builds the proper interface.

## Run
```
python -m pytest -q
python run.py
```
Open http://127.0.0.1:5000. The first start takes about 15 seconds while the model loads. To try from a phone on the same Wi-Fi: `python run.py --host 0.0.0.0` (anyone on that network can then reach it; use a trusted network only).

## Manual checks (record the outcome of each for Chapter Four)
1. A normal JPG leaf photograph: result page with category, confidence, explanation, disclaimer.
2. A PNG.
3. A .txt file, a text file renamed to .jpg, a damaged image, an empty submission.
4. A file larger than 5 MB.
5. A very small image (under 64 pixels).
6. Two files at once (the form allows one; test by a crafted request or skip).
7. Turn the model off (rename `models/resnet50_banana.keras`) and confirm the start-up error is clear.

## Limits
- Local single-user demonstration: no accounts, no CSRF protection, no HTTPS.
- Flask's built-in server is for demonstration only.
- Not yet adopted: the input check. To adopt only the threshold: `python -m src.thresholds --negatives data/negatives --adopt`. To adopt the input check as a warning as well: `python -m src.thresholds --negatives data/negatives --adopt-input-check`.
