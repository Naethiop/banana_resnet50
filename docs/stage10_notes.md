# Stage 10 Notes: Frontend

## What was built
- Pages: home with upload form (`/`), result page, error page. Plain HTML, one stylesheet, one small script. No framework, no Bootstrap, no external fonts, scripts or images: nothing is requested from other sites, so the interface works offline and sends nothing to third parties.
- Home page: title and purpose, upload form, preview of the chosen photograph before upload, photo tips, the five categories (from `app/content/conditions.json`, with potassium deficiency labelled a nutrient disorder, not an infectious disease), and "what this tool can and cannot do".
- Result page: the user's photograph, "Most likely category", the kind of condition, model confidence as a number and a bar (with a note that it is not the chance of being right), optional gentle notices, a description, what to do next, the next most likely categories, and a button to analyse another photograph. The disclaimer and the scope note appear on every page.
- Script (`app/static/js/main.js`): preview, early advice on file type and size (the server repeats every check), a loading indicator on submit, and restoring the page state when the browser's back button is used. Without JavaScript the form still works.
- Accessibility basics: language set, one main heading per page, labelled file input, alt text, skip link, announced loading and error messages, visible keyboard focus, buttons at least 44 px tall, colour contrast checked by calculation for the green button, and reduced-motion support for the spinner.
- Responsive: single column on phones, photograph beside text on wider screens.

## Content decisions to review (they affect what users read)
1. Performance statement on the home page: "In tests on photographs it had not seen, it chose the right category about 7 times out of 10, with wide uncertainty." Source: cross-validation accuracy 0.72 (Stage 6a). Change it if you prefer not to state a figure or if results change.
2. Photo tip: "Show the whole leaf or a large part of it... Very close-up photographs of a small part of a leaf were sometimes misjudged in our tests." Source: the Stage 7 error analysis (one photograph; weak evidence).
3. Descriptions of the five categories: based on Chapter Two (Alanazi, 2025). Please check them against your sources.
4. The warning for unsuitable images only appears if the input check is adopted (decision pending). The low-confidence warning uses the softer wording you chose.

## Testing done by the assistant
- Page structure, accessibility attributes, absence of external resources, static files, form without JavaScript: automated tests.
- The script's behaviour (type and size advice, preview, loading state, back-button restore) tested in Node with a stand-in browser.
- Not done: the layout has not been viewed in a real browser by the assistant (none was available). Please check it yourself.

## Manual checks on your side (record the outcome for Chapter Four)
1. Desktop browser: home page, choose a photograph, preview appears, analyse, result page.
2. Phone browser (same Wi-Fi, `python run.py --host 0.0.0.0`) or a narrow browser window: no sideways scrolling, text readable, buttons easy to tap.
3. Keyboard only: Tab reaches the file input and button, focus is visible, Enter submits.
4. Choose a .txt file, a photograph over 5 MB: advice appears at once and the button stays disabled.
5. Disconnect from the internet: the pages still load and work.
6. Send me a screenshot of the home page and of a result page.

## Deferred to Stage 11 (optional features, after this stage is reviewed)
Camera capture (FR8), prediction history (FR9, only if feasible), and adopting the input-check warning.

## Run
```
python -m pytest -q
python run.py
```
