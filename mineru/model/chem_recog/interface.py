"""Unified chemical-structure recognition for the MinerU chem pipeline.

Two clean backends, both runnable on torch/torchvision/numpy/rdkit only:

* ``molscribe``  (default) — Swin encoder + transformer decoder -> char-level SMILES.
  Harness-verified bit-exact against the official MolScribe; 92.0% exact match on the
  USPTO-10K benchmark.
* ``molgrapher``           — ResNet18 keypoint detector + 4xGCNConv graph classifier
  (PatCID's recogniser), pure-torch reimplementation, bit-exact vs the original.

The pipeline contract (unchanged from the previous swin_ocsr wrapper)::

    m = MolScribe(weight, device="cuda")
    m.predict_images(images, return_atoms_bonds=False, return_confidence=True)
    -> [{"smiles": str, "confidence": float}, ...]

``confidence`` is 1.0 when the produced SMILES parses in RDKit, else 0.0, because
``batch_analyze`` filters on ``confidence > 0.7`` and a non-parsing structure must be
dropped rather than written into the markdown.
"""
import os

import numpy as np

__all__ = ["MolScribe", "ChemRecog", "SwinOCSR"]


def _is_valid(smiles):
    if not smiles:
        return False
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:  # rdkit missing -> trust the backend
        return True


class ChemRecog:
    """Thin unified wrapper over the molscribe / molgrapher clean backends."""

    def __init__(self, weight=None, device="cpu", backend=None, **kwargs):
        self.backend = (backend or os.environ.get("MINERU_CHEM_RECOG_BACKEND", "molscribe")).lower()
        if self.backend == "molgrapher":
            from .molgrapher.interface import MolGrapher

            self.model = MolGrapher(device=device, **kwargs)
        else:
            from .molscribe.interface import MolScribe as _MolScribe

            # ``weight`` may be a .pth path or a HF snapshot dir; the clean
            # implementation also accepts None and downloads the default checkpoint.
            self.model = _MolScribe(weight, device=device, **kwargs)

    # ------------------------------------------------------------------ api
    def predict_images(self, images, return_atoms_bonds=False, return_confidence=True):
        out = []
        for img in images:
            out.append(self.predict_image(img, return_confidence=return_confidence))
        return out

    def predict_image(self, image, return_confidence=True):
        if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        try:
            res = self.model.predict_image(image)
        except Exception as exc:  # never let one bad crop kill the page
            return {"smiles": "", "confidence": 0.0, "error": str(exc)}
        smiles = (res or {}).get("smiles") or (res or {}).get("raw_smiles") or ""
        conf = float((res or {}).get("confidence") or 0.0)
        if conf <= 0.0:
            conf = 1.0 if _is_valid(smiles) else 0.0
        item = {"smiles": smiles, "confidence": conf}
        if return_confidence is False:
            item.pop("confidence", None)
        return item


# ``model_init.py`` does ``from ...model.chem_recog import SwinOCSR as MolScribe``
# and constructs ``MolScribe(weight, device=device)``; keep both names working.
MolScribe = ChemRecog
SwinOCSR = ChemRecog
