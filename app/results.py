"""Turns a prediction into what the user sees: names, explanations and gentle warnings. No Flask here."""
import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

CONTENT = Path(__file__).resolve().parent / "content"

DISCLAIMER = ("This tool gives a preliminary indication from a photograph only. It does not replace laboratory "
              "diagnosis or the advice of an agricultural extension worker or other qualified plant health professional, "
              "and it should not be the only basis for decisions such as applying pesticides.")
SCOPE_NOTE = ("The system recognises banana leaf photographs in five categories only. For any other image, including "
              "other plants, it will still choose one of the five, sometimes with high confidence.")
LOW_CONFIDENCE_MESSAGE = ("The system is less certain about this image. A clearer photograph of a single leaf in good "
                          "light may help, and the result is best confirmed by an agricultural extension worker.")
UNSUITABLE_MESSAGE = ("This image may not be a banana leaf photograph, or it may look different from the photographs the "
                      "system learned from. The result may be less reliable.")
CONFIDENCE_NOTE = "Confidence describes how strongly the model favours this category. It is not the probability that the result is correct."


def load_conditions():
    return json.loads((CONTENT / "conditions.json").read_text(encoding="utf-8"))


def build_view(result, conditions):
    """result: dict from Predictor.predict. Returns a plain dict for the template."""
    info = conditions.get(result["label"], {"name": result["label"].replace("-", " "), "kind": "category",
                                             "description": "", "advice": ""})
    notices = []
    if result.get("unsuitable"):
        notices.append(UNSUITABLE_MESSAGE)
    if result.get("low_confidence"):
        notices.append(LOW_CONFIDENCE_MESSAGE)
    other = [{"name": conditions.get(n, {}).get("name", n.replace("-", " ")), "percent": round(100 * p)}
             for n, p in result["top"][1:]]
    return {"name": info["name"], "kind": info["kind"], "description": info["description"], "advice": info["advice"],
            "percent": round(100 * result["confidence"]), "other": other, "notices": notices,
            "seconds": round(result.get("seconds", 0.0), 2)}


def thumbnail_data_uri(data, max_side=480):
    """A small JPEG of the upload as a data URI, so the page can show it without storing any file."""
    with Image.open(BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = BytesIO()
        im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
