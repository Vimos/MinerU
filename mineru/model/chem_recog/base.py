"""Shared base class + factory for the unified OCSR package.

Runtime dependencies of the whole package: torch, torchvision, numpy, rdkit,
PIL, huggingface_hub.  Anything else (timm, OpenNMT, albumentations,
pytorch_lightning, torch_geometric, mol_depict, cv2, mahotas) is deliberately
absent - both backends are re-implementations in plain torch/numpy.
"""
import numpy as np

__all__ = ["OCSRBase", "load_ocsr", "BACKENDS"]

_ALIASES = {
    "molscribe": "molscribe",
    "ms": "molscribe",
    "scribe": "molscribe",
    "molgrapher": "molgrapher",
    "mg": "molgrapher",
    "grapher": "molgrapher",
}


class OCSRBase:
    """Common interface: ``predict_image(image) -> {"smiles": str, ...}``.

    ``image`` is either a ``HxWx3`` uint8 RGB numpy array or a PIL image; the
    returned dict always contains at least ``smiles`` (may be ``None`` when the
    backend could not build a molecule).
    """

    name = "ocsr"
    device = None

    def predict_image(self, image, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError

    # -- convenience ---------------------------------------------------- #
    @staticmethod
    def to_numpy_rgb(image):
        """Accept PIL / HxW / HxWx3 / HxWx4 / float arrays -> HxWx3 uint8 RGB."""
        from PIL import Image

        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"))
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[..., :3]
        if image.dtype != np.uint8:
            scale = 255.0 if image.max() <= 1.0 else 1.0
            image = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    def predict_batch(self, images, **kwargs):
        return [self.predict_image(img, **kwargs) for img in images]

    def predict_smiles(self, image, **kwargs):
        out = self.predict_image(image, **kwargs)
        return out.get("smiles")

    @property
    def load_report(self):  # backends override with a dict of key reports
        return getattr(self, "_load_report", None)

    @load_report.setter
    def load_report(self, value):
        self._load_report = value

    def __repr__(self):
        return f"<{type(self).__name__} device={self.device}>"


BACKENDS = ("molscribe", "molgrapher")


def load_ocsr(name="molscribe", **kwargs):
    """Factory: ``load_ocsr("molscribe"|"molgrapher", **kwargs)``."""
    key = str(name).strip().lower()
    if key not in _ALIASES:
        raise ValueError(f"unknown OCSR backend {name!r}; choose from {BACKENDS}")
    key = _ALIASES[key]

    if key == "molscribe":
        from .molscribe import MolScribe

        return MolScribe(**kwargs)

    from .molgrapher import MolGrapher

    return MolGrapher(**kwargs)
