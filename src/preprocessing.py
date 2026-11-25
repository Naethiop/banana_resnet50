"""The single image-preprocessing definition used by training, evaluation and the application.

Steps: honour EXIF orientation, convert to RGB, centre-crop to a square, resize to 224 x 224,
then apply the preprocessing required by the Keras ResNet50 ImageNet weights ("caffe" mode:
RGB -> BGR and subtraction of the ImageNet channel means). The dataset images are already square,
so the crop changes nothing for them; it only prevents distortion of non-square photographs.
"""
import numpy as np
from PIL import Image, ImageOps

from src import config

# Channel means for BGR order, as used by keras.applications.resnet50.preprocess_input
IMAGENET_BGR_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def _centre_crop_square(im):
    w, h = im.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return im.crop((left, top, left + s, top + s))


def load_rgb(source, size=config.IMAGE_SIZE):
    """Load a path or file-like object as a float32 array of shape (224, 224, 3), RGB, values 0..255."""
    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        im = _centre_crop_square(im).resize(size, Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)


def preprocess_batch(batch):
    """Apply the ResNet50 ImageNet preprocessing to an array of RGB images (0..255)."""
    x = np.asarray(batch, dtype=np.float32)
    x = x[..., ::-1]                      # RGB -> BGR
    return x - IMAGENET_BGR_MEAN          # new array; the input is not modified
