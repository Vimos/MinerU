"""Inference preprocessing, reproducing MolScribe's ``get_transforms(384)``.

MolScribe's validation/inference pipeline (``dataset.get_transforms`` with
``augment=False``) is exactly::

    CropWhite(pad=5) -> Resize(384, 384) -> ToGray(p=1)
    -> Normalize(ImageNet mean/std) -> ToTensor

i.e. RGB in, **grayscale** out, resized to 384x384 and ImageNet-normalised.
Implemented here with numpy + PIL only (no albumentations / OpenCV).
"""
import numpy as np
from PIL import Image

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
WHITE = (255, 255, 255)


def crop_white(img, pad=5, value=WHITE):
    """Crop to the bbox of non-white pixels, then pad ``pad`` white pixels.

    Mirrors ``molscribe/augment.py::CropWhite``.
    """
    assert img.ndim == 3 and img.shape[2] == 3
    diff = (img != np.array(value, dtype=img.dtype)).any(axis=2)
    if not diff.any():
        return img
    rows = np.where(diff.any(axis=1))[0]
    cols = np.where(diff.any(axis=0))[0]
    out = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    if pad > 0:
        h, w = out.shape[:2]
        canvas = np.full((h + 2 * pad, w + 2 * pad, out.shape[2]),
                         np.asarray(value, dtype=out.dtype), dtype=out.dtype)
        canvas[pad:pad + h, pad:pad + w] = out
        out = canvas
    return np.ascontiguousarray(out)


def rgb_to_gray(img):
    """cv2.cvtColor(img, COLOR_RGB2GRAY) equivalent (ITU-R BT.601)."""
    img = img.astype(np.float32)
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def preprocess_image(img, input_size=384):
    """RGB uint8 ``(H, W, 3)`` -> float32 tensor ``(3, input_size, input_size)``."""
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[..., :3]
    img = crop_white(img, pad=5)
    pil = Image.fromarray(img.astype(np.uint8)).resize(
        (input_size, input_size), Image.BILINEAR)
    img = np.asarray(pil)
    gray = rgb_to_gray(img)
    img = np.stack([gray] * 3, axis=-1).astype(np.float32) / 255.0
    img = (img - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return np.ascontiguousarray(img.transpose(2, 0, 1))
