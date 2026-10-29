"""MolGrapher image preprocessing (numpy + PIL only, no cv2 / mol_depict).

Reproduces ``molgrapher/datasets/dataset_image.py`` with the PaddleOCR caption
remover disabled::

    resize_image(1024, border=30)         # aspect preserving, white canvas
    [caption removal - PaddleOCR, skipped]
    crop_tight()                          # bbox of the ink + min 60px window
    resize_image(1024, border=90)         # border_size * 3
    binarize (>0.6 -> 1, else 0), 3 channels, invert  (ink = 0, paper = 1)

``resize_image`` is a verbatim port of
``mol_depict.utils.utils_image.resize_image``.
"""
import numpy as np
import torch
from PIL import Image, ImageOps

__all__ = ["resize_image", "crop_tight", "preprocess_image", "to_tensor",
           "binarize"]


def resize_image(image, image_size, border_size=30, b="white"):
    """Aspect-preserving resize + square white padding + border padding."""
    width = image_size[0] - 2 * border_size
    height = image_size[1] - 2 * border_size
    width_ratio = width / image.width
    height_ratio = height / image.height

    if width_ratio < height_ratio:
        transformed_width = width
        transformed_height = round(width_ratio * image.height)
    else:
        transformed_width = round(height_ratio * image.width)
        transformed_height = height

    transformed_image = image.resize((transformed_width, transformed_height),
                                     Image.LANCZOS)
    fill = 0 if b == "black" else 255
    background = Image.new("L", (width, height), fill)
    offset = (round((width - transformed_width) / 2),
              round((height - transformed_height) / 2))
    background.paste(transformed_image, offset)
    return ImageOps.expand(background, border=border_size,
                           fill="black" if b == "black" else "white")


def crop_tight(pil_image, dilate=False):
    im = pil_image.copy()
    bbox = ImageOps.invert(im).getbbox()
    if bbox is None:
        return pil_image
    min_cropping_size = 60
    if abs(bbox[2] - bbox[0]) > min_cropping_size or \
            abs(bbox[3] - bbox[1]) > min_cropping_size:
        return pil_image.crop(bbox)
    missing_x = min_cropping_size - abs(bbox[2] - bbox[0])
    missing_y = min_cropping_size - abs(bbox[3] - bbox[1])
    bbox = [bbox[0] - missing_x // 2, bbox[1] - missing_y // 2,
            bbox[2] + missing_x // 2, bbox[3] + missing_y // 2]
    return pil_image.crop(bbox)


def preprocess_image(image, image_size=1024, border_size=30):
    """HxWx3 uint8 RGB (or PIL) -> PIL 'L' 1024x1024, white background."""
    if not isinstance(image, Image.Image):
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 4:
            arr = arr[..., :3]
        image = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    else:
        image = image.convert("RGB")

    pil = resize_image(image, (image_size, image_size), border_size=border_size,
                       b="white")
    # caption removal (PaddleOCR) intentionally skipped - see module docstring
    pil = Image.fromarray(np.stack([np.array(pil)] * 3, axis=-1))
    pil = crop_tight(pil)
    return resize_image(pil, (image_size, image_size), border_size=border_size * 3)


def binarize(pil_image):
    """(H, W, 3) float32 in {0,1}: ink -> 0, paper -> 1 (upstream threshold)."""
    if isinstance(pil_image, Image.Image) and pil_image.mode != "L":
        pil_image = pil_image.convert("L")
    image = np.array(pil_image, dtype=np.float32)
    if image.ndim == 3:
        image = image[..., 0]
    image = image / 255
    image[image > 0.6] = 1.0
    image[image != 1.0] = 0.0
    return np.stack((image,) * 3, axis=-1)


def to_tensor(pil_image):
    """PIL 'L' -> (1, 3, H, W) float tensor, inverted binary (0=ink, 1=paper)."""
    arr = torch.from_numpy(np.ascontiguousarray(binarize(pil_image))).permute(2, 0, 1)
    return (1.0 - arr).unsqueeze(0)
