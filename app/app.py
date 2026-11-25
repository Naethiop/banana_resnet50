"""Flask application: routes, error handlers and start-up. Contains no machine-learning logic."""
import logging
from io import BytesIO
from pathlib import Path

from urllib.parse import urlparse

from flask import Flask, abort, redirect, render_template, request, url_for

from app import history
from app.results import (CONFIDENCE_NOTE, DISCLAIMER, SCOPE_NOTE, build_view, load_conditions,
                         thumbnail_data_uri)
from app.validators import ValidationError, read_and_validate
from src import config

log = logging.getLogger("banana_app")


def create_app(predictor=None, history_path=None):
    """predictor: an object with .predict(file_like); loaded from the model package if not given.
    history_path: SQLite file for FR9; defaults to config.HISTORY_DB when config.ENABLE_HISTORY is true."""
    here = Path(__file__).resolve().parent
    app = Flask(__name__, template_folder=str(here / "templates"), static_folder=str(here / "static"))
    # headroom for the multipart envelope; the exact 5 MB limit is enforced by the validator
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024 + 256 * 1024
    if predictor is None:
        from src.predict import Predictor
        predictor = Predictor()
    conditions = load_conditions()
    history_db = history_path if history_path is not None else (config.HISTORY_DB if config.ENABLE_HISTORY else None)
    if history_db:
        history.init(history_db)
    shared = {"disclaimer": DISCLAIMER, "scope_note": SCOPE_NOTE, "max_mb": config.MAX_UPLOAD_MB,
              "max_side": config.CLIENT_MAX_SIDE, "history_enabled": bool(history_db)}
    categories = [{"name": v["name"], "kind": v["kind"], "description": v["description"]}
                  for v in conditions.values()]

    @app.after_request
    def headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/")
    def index():
        return render_template("index.html", categories=categories, **shared)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def predict():
        files = request.files.getlist("image")
        try:
            if len(files) > 1:
                raise ValidationError("Please upload one image at a time.")
            data = read_and_validate(files[0] if files else None)
        except ValidationError as err:
            return render_template("index.html", error=str(err), categories=categories, **shared), 400
        result = predictor.predict(BytesIO(data))                 # in memory only: no file is ever written
        view = build_view(result, conditions)
        if history_db:
            try:
                history.add(history_db, result)               # text only; a failure here never blocks the result
            except Exception:  # noqa: BLE001
                log.exception("Could not save the prediction history")
        return render_template("result.html", view=view, thumb=thumbnail_data_uri(data),
                               confidence_note=CONFIDENCE_NOTE, **shared)

    if history_db:
        @app.get("/history")
        def show_history():
            rows = history.recent(history_db)
            for r in rows:
                r["name"] = conditions.get(r["category"], {}).get("name", r["category"].replace("-", " "))
                r["percent"] = round(100 * r["confidence"])
            return render_template("history.html", rows=rows, **shared)

        @app.post("/history/clear")
        def clear_history():
            origin = request.headers.get("Origin")
            if origin and urlparse(origin).netloc != request.host:        # refuse requests from other web sites
                abort(400)
            history.clear(history_db)
            return redirect(url_for("show_history"))

    @app.errorhandler(413)
    def too_large(_):
        return render_template("index.html", error=f"The upload is larger than {config.MAX_UPLOAD_MB} MB. "
                               "Please upload a smaller image.", categories=categories, **shared), 413

    @app.errorhandler(404)
    def not_found(_):
        return render_template("error.html", title="Page not found", message="That page does not exist.",
                               **shared), 404

    @app.errorhandler(405)
    def not_allowed(_):
        return render_template("error.html", title="Not allowed", message="That action is not allowed here.",
                               **shared), 405

    @app.errorhandler(Exception)
    def server_error(exc):
        if hasattr(exc, "code") and isinstance(exc.code, int) and exc.code < 500:
            return exc
        log.exception("Unhandled error while handling a request")      # no image data or file names are logged
        return render_template("error.html", title="Something went wrong",
                               message="The image could not be analysed. Please try again with another photograph.",
                               **shared), 500

    return app
