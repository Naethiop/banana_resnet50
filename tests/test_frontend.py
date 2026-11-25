"""Frontend checks: rendered pages, accessibility basics, no external requests, and the script's behaviour."""
import re
import shutil
import subprocess
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from app.app import create_app
from tests.test_app import FakePredictor, image_bytes, post


def client(**kw):
    return create_app(FakePredictor(**kw)).test_client()


def test_home_page_structure_and_accessibility():
    t = client().get("/").get_data(as_text=True)
    assert '<html lang="en">' in t and 'name="viewport"' in t
    assert 'for="image"' in t and 'id="image"' in t and 'enctype="multipart/form-data"' in t
    assert 'role="status"' in t and 'aria-live="polite"' in t            # loading message is announced
    assert 'alt="Preview of the photograph you chose"' in t
    assert "Skip to the main content" in t
    assert t.count("<h1") == 1                                            # one main heading


def test_home_page_content_matches_decisions():
    t = client().get("/").get_data(as_text=True)
    for name in ("Black Sigatoka", "Yellow Sigatoka", "Panama disease", "Potassium deficiency", "Healthy leaf"):
        assert name in t
    assert "nutrient disorder" in t and "not an infectious disease" in t
    assert "banana leaf condition" in t and "classification category" in t
    assert "does not replace laboratory" in t and "five categories only" in t
    assert "not</strong> the chance" in t


def test_result_page_accessibility_and_meter():
    t = post(client(), image_bytes()).get_data(as_text=True)
    assert 'alt="The photograph you uploaded"' in t and 'role="img"' in t and "Model confidence" in t
    assert 'width: 90%' in t and "Analyse another photograph" in t


def test_no_external_resources_are_loaded():
    for page in (client().get("/").get_data(as_text=True), post(client(), image_bytes()).get_data(as_text=True)):
        assert not re.search(r'(src|href)="https?://', page)             # works offline; nothing is sent to third parties


def test_static_files_are_served():
    c = client()
    assert c.get("/static/css/style.css").status_code == 200 and c.get("/static/js/main.js").status_code == 200


def test_form_works_without_javascript():
    t = client().get("/").get_data(as_text=True)
    assert "disabled" not in re.search(r'<button[^>]*id="analyse"[^>]*>', t).group(0)   # script disables it, not the HTML


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_script_behaviour_in_node():
    script = Path(__file__).parent / "js" / "test_main.js"
    r = subprocess.run(["node", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr or r.stdout


def test_camera_input_is_not_submitted_and_has_capture_hint():
    t = client().get("/").get_data(as_text=True)
    cam = re.search(r'<input[^>]*id="camera"[^>]*>', t).group(0)
    assert 'capture="environment"' in cam and "name=" not in cam          # not part of the form data
    assert 'id="camera-button"' in t and re.search(r'<button[^>]*id="camera-button"[^>]*hidden', t)
    assert t.count('name="image"') == 1                                    # only one image field is submitted
