"""Upload validation. Everything is checked on the server; browser checks are only a convenience."""
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src import config

Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS
MAX_BYTES = config.MAX_UPLOAD_MB * 1024 * 1024
ACCEPTED = "JPG, JPEG or PNG"


class ValidationError(Exception):
    """A problem with the upload that the user can fix. The message is shown to the user."""


def sniff(data):
    """Detect the real file type from its first bytes."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return None


def read_and_validate(file_storage):
    """Return the image bytes of one valid upload, or raise ValidationError. Nothing is written to disk."""
    if file_storage is None or not file_storage.filename:
        raise ValidationError(f"No file was selected. Please choose one banana leaf photograph ({ACCEPTED}).")
    ext = Path(file_storage.filename).suffix.lower().lstrip(".")
    if ext not in config.ALLOWED_EXTENSIONS:
        raise ValidationError(f"This file type is not supported. Please upload a {ACCEPTED} image.")
    data = file_storage.stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValidationError(f"The file is larger than {config.MAX_UPLOAD_MB} MB. Please upload a smaller image.")
    if not data:
        raise ValidationError("The file is empty. Please choose another image.")
    if sniff(data) is None:
        raise ValidationError(f"The file is not a valid {ACCEPTED} image. Please choose another file.")
    try:
        with Image.open(BytesIO(data)) as im:
            im.verify()
        with Image.open(BytesIO(data)) as im:
            width, height = im.size
            im.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise ValidationError("The image could not be read. It may be damaged; please choose another one.")
    if min(width, height) < config.MIN_IMAGE_SIDE:
        raise ValidationError(f"The image is too small ({width} x {height} pixels). "
                              f"Please use a photograph at least {config.MIN_IMAGE_SIDE} pixels on each side.")
    return data
