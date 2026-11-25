"""FR9: text-only prediction history (Stage 11)."""
import sqlite3
import tempfile
from pathlib import Path

from app.app import create_app
from tests.test_app import FakePredictor, image_bytes, post


def fresh_client(**kw):
    db = Path(tempfile.mkdtemp()) / "h.sqlite"
    return create_app(FakePredictor(**kw), history_path=db).test_client(), db


def test_prediction_is_recorded_and_listed():
    c, db = fresh_client(low=True)
    assert post(c, image_bytes(), "secret-name.jpg").status_code == 200
    page = c.get("/history").get_data(as_text=True)
    assert "Black Sigatoka" in page and "90%" in page and "Less certain" in page
    assert "secret-name" not in page


def test_only_text_fields_are_stored_no_photo_or_file_name():
    c, db = fresh_client()
    post(c, image_bytes(), "secret-name.jpg")
    con = sqlite3.connect(db)
    cols = [r[1] for r in con.execute("PRAGMA table_info(predictions)")]
    assert cols == ["id", "created_at", "category", "confidence", "low_confidence", "unsuitable"]
    raw = db.read_bytes()
    assert b"secret-name" not in raw and b"\xff\xd8\xff" not in raw and b"JFIF" not in raw
    con.close()


def test_failed_validation_is_not_recorded_and_clear_works():
    c, db = fresh_client()
    post(c, b"nope", "x.txt")
    assert "No results have been saved yet." in c.get("/history").get_data(as_text=True)
    post(c, image_bytes())
    assert "Black Sigatoka" in c.get("/history").get_data(as_text=True)
    r = c.post("/history/clear")
    assert r.status_code == 302
    assert "No results have been saved yet." in c.get("/history").get_data(as_text=True)


def test_clear_refuses_requests_from_other_sites():
    c, db = fresh_client()
    post(c, image_bytes())
    r = c.post("/history/clear", headers={"Origin": "http://evil.example"})
    assert r.status_code == 400
    assert "Black Sigatoka" in c.get("/history").get_data(as_text=True)


def test_history_failure_never_blocks_the_result():
    import app.history as h
    original = h.add
    try:
        h.add = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full"))
        c, db = fresh_client()
        r = post(c, image_bytes())
        assert r.status_code == 200 and "Black Sigatoka" in r.get_data(as_text=True)
    finally:
        h.add = original


def test_history_link_and_page_use_the_shared_layout():
    c, db = fresh_client()
    home = c.get("/").get_data(as_text=True)
    assert 'href="/history"' in home and "does not replace laboratory" in c.get("/history").get_data(as_text=True)
