# Stage 11 Notes: Optional Features and Integration

## Features
- **FR8 camera capture:** a "Take a photo with the camera" button, shown only on touch devices (where `capture="environment"` opens the camera). On a computer, choose a file instead; webcam capture on desktop is not supported. The camera photograph is placed into the normal upload field, so the server sees one image exactly as before.
- **Large photographs:** phone photographs are often over 5 MB. The page now shrinks photographs larger than 1 MB (longest side 1600 pixels, JPEG) in the browser before upload. The model works at 224 pixels, so nothing the model uses is lost. If shrinking is not possible and the file is over 5 MB, the user is told. The server still enforces 5 MB.
- **FR7 low-confidence flow:** the gentle notice from the backend appears on the result page when the model confidence is below the adopted threshold (0.94 once adopted). It is also noted in the history.
- **Input-check warning:** appears only if the similarity check has been adopted (pending the student's decision). Evidence and limits are in the Stage 8 and 9 notes.
- **FR9 prediction history (SQLite, text only):** category, confidence, time, and the two flags. No photographs, no file names. Page `/history`, link in the header, and a "Clear all history" button (refuses requests from other web sites). The list is shared by everyone who uses the computer. A failure to save never blocks a result. To switch it off, set `ENABLE_HISTORY = False` in `src/config.py`. The database is `app/instance/history.sqlite` (not committed to Git).
- `python -m src.predict --status` shows whether the threshold and the input check are adopted, without loading the model.

## Decisions pending from the student
1. Input check: `python -m src.thresholds --negatives data/negatives --adopt-input-check` (yes) or leave it off (no).
2. Low-confidence threshold: `python -m src.thresholds --negatives data/negatives --adopt` (approved, with the softer wording). Check the result with `python -m src.predict --status`.

## Run
```
python -m pytest -q
python -m src.predict --status
python run.py
```

## Manual checks (record the outcomes for Chapter Four)
1. `--status` shows the low-confidence threshold as adopted (0.94). If it says NOT adopted, run the `--adopt` command first.
2. Analyse a photograph, then open History: the entry shows category, confidence and time only.
3. Analyse a very blurry or unusual photograph: if the confidence is below the threshold, the gentle notice appears and History says "Less certain".
4. Clear the history and check that the list is empty.
5. On a phone (same Wi-Fi, `python run.py --host 0.0.0.0`, then the computer's address in the phone browser): the camera button appears; take a photograph; it is previewed and analysed. Try a full-resolution photograph (over 5 MB).
6. On a computer: the camera button does not appear.
7. Inspect `app/instance/history.sqlite` with any SQLite viewer (optional): only the six text columns exist.

## Limits
- Camera capture depends on the phone browser; some browsers show a chooser instead of opening the camera directly.
- The browser-side shrinking relies on current browser features; where they are missing the original file is used and the 5 MB limit applies.
- History is a convenience on a single-user local demonstration. It has no accounts.
