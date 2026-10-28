"""SwinOCSR wrapper: chemical-structure image -> SMILES.

Clean reimplementation on top of the user's ReactiGraph swin_ocsr model.
Deps: torch, torchvision, transformers, deepsmiles  (no albumentations/timm/OpenNMT).
"""
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .swin_ocsr import SwinOCSRConfig, SwinOCSRModel  # registers with Auto*

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swin_ocsr")
_WORD_MAP = os.path.join(_MODEL_DIR, "500wan_shuffle_DeepSMILES_word_map")

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _subsequent_mask(size):
    attn_shape = (1, size, size)
    m = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(m) == 0


def _deepsmiles_to_smiles(deep_smiles):
    import deepsmiles
    converter = deepsmiles.Converter(rings=True, branches=True)
    try:
        return converter.decode(deep_smiles)
    except Exception:  # noqa: BLE001
        return None


class SwinOCSR:
    """Drop-in for the old MolScribe interface used by batch_analyze."""

    def __init__(self, weight, device="cpu"):
        from transformers import AutoModel  # local import to keep module import light
        self.device = torch.device(device if torch.cuda.is_available() or str(device).startswith("cpu") else "cpu")
        self.model = SwinOCSRModel.from_pretrained(weight) if os.path.isdir(weight) else SwinOCSRModel(SwinOCSRConfig())
        self.model.to(self.device).eval()
        self.word_map = torch.load(_WORD_MAP, map_location="cpu")
        self.inv_word_map = {v: k for k, v in self.word_map.items()}
        self.start_id = self.word_map["<start>"]
        self.end_id = self.word_map["<end>"]
        self.max_len = 102

    @torch.no_grad()
    def _predict_one(self, img_np):
        pil = Image.fromarray(img_np).convert("RGB")
        x = _transform(pil).unsqueeze(0).to(self.device)
        memory = self.model.encoder(x)
        ys = torch.ones(1, 1).fill_(self.start_id).type(torch.long).to(self.device)
        logprobs = []
        for _ in range(self.max_len - 1):
            logit = self.model.decoder(ys, memory, _subsequent_mask(ys.size(1)).type(torch.long).to(self.device))
            prob = nn.Softmax(dim=-1)(logit[:, -1])
            p, nxt = torch.max(prob, dim=-1)
            logprobs.append(float(p[0]))
            nxt = nxt.unsqueeze(1)
            ys = torch.cat([ys, nxt.type(torch.long).to(self.device)], dim=1)
            if int(nxt[0, 0]) == self.end_id:
                break
        tokens = [t for t in ys[0, 1:].tolist() if t != self.end_id]
        deep_smiles = "".join(self.inv_word_map.get(t, "") for t in tokens)
        smiles = _deepsmiles_to_smiles(deep_smiles)
        confidence = float(np.mean(logprobs)) if logprobs else 0.0
        return smiles, confidence

    def predict_images(self, images, return_atoms_bonds=False, return_confidence=True):
        out = []
        for img in images:
            smiles, conf = self._predict_one(img)
            out.append({"smiles": smiles, "confidence": conf})
        return out


# Backwards-compat alias (model_init imports MolScribe by this name).
MolScribe = SwinOCSR
