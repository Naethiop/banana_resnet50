"""Backend tests with a fake predictor (no TensorFlow needed): validation, privacy, messages and errors."""
import os
from io import BytesIO

from PIL import Image

from app.app import create_app
from app.results import LOW_CONFIDENCE_MESSAGE, UNSUITABLE_MESSAGE


class FakePredictor:
    def __init__(self, low=None, unsuitable=None, boom=False):
        self.low, self.unsuitable, self.boom, self.seen = low, unsuitable, boom, []

    def predict(self, source):
        if self.boom:
            raise RuntimeError("secret internal detail")
        self.seen.append(type(source).__name__)
        return {"label": "Black-Sigatoka", "confidence": 0.9, "top": [("Black-Sigatoka", 0.9), ("Yellow-Sigatoka", 0.07), ("Healthy-Leaf", 0.03)],
                "low_confidence": self.low,
                "unsuitable": self.unsuitable, "seconds": 0.5}


def image_bytes(size=(200, 200), fmt="JPEG"):
    buf = BytesIO()
    Image.new("RGB", size, (30, 120, 40)).save(buf, fmt)
    return buf.getvalue()


def post(client, data, name="leaf.jpg", field="image"):
    return client.post("/predict", data={field: (BytesIO(data), name)}, content_type="multipart/form-data")


def test_home_page_has_disclaimer_and_scope():
    r = create_app(FakePredictor()).test_client().get("/")
    t = r.get_data(as_text=True)
    assert r.status_code == 200 and "does not replace laboratory" in t and "five categories only" in t


def test_valid_upload_shows_category_confidence_and_stays_in_memory():
    fake = FakePredictor()
    r = post(create_app(fake).test_client(), image_bytes())
    t = r.get_data(as_text=True)
    assert r.status_code == 200 and "Black Sigatoka" in t and "90%" in t and "does not replace laboratory" in t
    assert fake.seen == ["BytesIO"]                      # the predictor received an in-memory object, not a file path
    assert "data:image/jpeg;base64," in t               # the preview is embedded, not stored


def test_png_is_accepted():
    assert post(create_app(FakePredictor()).test_client(), image_bytes(fmt="PNG"), "leaf.png").status_code == 200


def test_missing_file_wrong_type_fake_image_and_tiny_image_are_rejected_with_clear_messages():
    c = create_app(FakePredictor()).test_client()
    r = c.post("/predict", data={}, content_type="multipart/form-data")
    assert r.status_code == 400 and "No file was selected" in r.get_data(as_text=True)
    r = post(c, b"hello", "notes.txt")
    assert r.status_code == 400 and "not supported" in r.get_data(as_text=True)
    r = post(c, b"this is not an image at all", "leaf.jpg")
    assert r.status_code == 400 and "not a valid" in r.get_data(as_text=True)
    r = post(c, image_bytes((10, 10)), "tiny.jpg")
    assert r.status_code == 400 and "too small" in r.get_data(as_text=True)
    r = post(c, b"", "empty.jpg")
    assert r.status_code == 400


def test_corrupt_jpeg_is_rejected():
    data = image_bytes()
    r = post(create_app(FakePredictor()).test_client(), data[:60] + b"\x00" * 50, "bad.jpg")
    assert r.status_code == 400 and "damaged" in r.get_data(as_text=True) or "not a valid" in r.get_data(as_text=True)


def test_file_over_five_megabytes_is_rejected_politely():
    big = b"\xff\xd8\xff" + os.urandom(5 * 1024 * 1024 + 1000)
    r = post(create_app(FakePredictor()).test_client(), big, "big.jpg")
    assert r.status_code in (400, 413) and "5 MB" in r.get_data(as_text=True)


def test_two_files_are_rejected():
    c = create_app(FakePredictor()).test_client()
    r = c.post("/predict", data={"image": [(BytesIO(image_bytes()), "a.jpg"), (BytesIO(image_bytes()), "b.jpg")]},
               content_type="multipart/form-data")
    assert r.status_code == 400 and "one image at a time" in r.get_data(as_text=True)


def test_gentle_warnings_appear_only_when_flagged():
    t = post(create_app(FakePredictor()).test_client(), image_bytes()).get_data(as_text=True)
    assert LOW_CONFIDENCE_MESSAGE not in t and UNSUITABLE_MESSAGE not in t
    t = post(create_app(FakePredictor(low=True, unsuitable=True)).test_client(), image_bytes()).get_data(as_text=True)
    assert LOW_CONFIDENCE_MESSAGE in t and UNSUITABLE_MESSAGE in t
    assert "wrong" not in LOW_CONFIDENCE_MESSAGE.lower()          # softer wording: does not say the result is wrong


def test_server_error_hides_details():
    r = post(create_app(FakePredictor(boom=True)).test_client(), image_bytes())
    t = r.get_data(as_text=True)
    assert r.status_code == 500 and "secret internal detail" not in t and "Traceback" not in t
    assert "could not be analysed" in t


def test_unknown_page_and_headers():
    c = create_app(FakePredictor()).test_client()
    r = c.get("/nothing-here")
    assert r.status_code == 404 and "does not exist" in r.get_data(as_text=True)
    h = c.get("/").headers
    assert h["X-Content-Type-Options"] == "nosniff" and h["Cache-Control"] == "no-store"


def test_no_files_are_written_by_a_prediction(tmp_path=None):
    import tempfile, pathlib
    before = {p for p in pathlib.Path(tempfile.gettempdir()).glob("*")}
    post(create_app(FakePredictor()).test_client(), image_bytes())
    after = {p for p in pathlib.Path(tempfile.gettempdir()).glob("*")}
    assert not [p for p in after - before if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
