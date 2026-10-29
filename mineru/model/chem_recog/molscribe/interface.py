"""High-level MolScribe interface: checkpoint loading + image -> SMILES."""
import os

import numpy as np
import torch

from ..base import OCSRBase
from .model import MolScribeNet
from .tokenizer import CharTokenizer
from .preprocess import preprocess_image
from .chemistry import convert_graph_to_smiles

CKPT_REPO = "yujieq/MolScribe"
CKPT_FILE = "swin_base_char_aux_1m680k.pth"
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VOCAB = os.path.join(_HERE, "vocab", "vocab_chars.json")


def download_checkpoint(repo=CKPT_REPO, filename=CKPT_FILE, local_dir=None):
    """Fetch the official checkpoint (local copy first, then the HF cache/mirror)."""
    if local_dir:
        local = os.path.join(local_dir, filename)
        if os.path.exists(local):
            return local
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename)


def strip_module_prefix(state_dict):
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


class MolScribe(OCSRBase):
    """Clean re-implementation of ``molscribe.interface.MolScribe``."""

    name = "molscribe"

    def __init__(self, model_path=None, vocab_path=None, device=None, input_size=384,
                 coord_bins=64, sep_xy=True):
        if model_path is None:
            model_path = download_checkpoint(local_dir=_HERE)
        if vocab_path is None:
            vocab_path = DEFAULT_VOCAB
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        states = torch.load(model_path, map_location="cpu", weights_only=True)
        self.checkpoint_args = states.get("args", {})

        self.tokenizer = CharTokenizer(input_size=coord_bins, path=vocab_path,
                                       sep_xy=sep_xy)
        self.vocab_size = len(self.tokenizer)
        self.input_size = input_size

        self.model = MolScribeNet(
            img_size=input_size,
            encoder_dim=1024,
            embed_dim=128,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
            window_size=12,
            dec_hidden_size=256,
            dec_num_layers=6,
            dec_attn_heads=8,
            vocab_size=self.vocab_size,
        )
        self.load_report = self.load_weights(states)
        self.model.to(self.device).eval()

    # ------------------------------------------------------------------ #
    def load_weights(self, states):
        """Strictly load the two sub-state-dicts; report every key mismatch."""
        enc_sd = strip_module_prefix(states["encoder"])
        dec_sd = strip_module_prefix(states["decoder"])

        report = {}
        for name, module, sd in (("encoder", self.model.encoder, enc_sd),
                                 ("decoder", self.model.decoder, dec_sd)):
            res = module.load_state_dict(sd, strict=False)
            report[name] = {"missing": list(res.missing_keys),
                            "unexpected": list(res.unexpected_keys)}
            if not res.missing_keys and not res.unexpected_keys:
                # proof: strict load of the very same dict also succeeds
                module.load_state_dict(sd, strict=True)
                report[name]["strict_ok"] = True
            else:
                report[name]["strict_ok"] = False
        report["n_encoder_keys"] = len(enc_sd)
        report["n_decoder_keys"] = len(dec_sd)
        return report

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_image(self, image, return_graph=False, max_length=480):
        """image: HxWx3 uint8 RGB numpy array (or PIL image).

        Returns dict with ``smiles`` (final RDKit-canonicalised SMILES built from
        atoms + bonds) and ``raw_smiles`` (the decoder's token string).
        """
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        tensor = torch.from_numpy(preprocess_image(image, self.input_size)).unsqueeze(0)
        tensor = tensor.to(self.device)

        preds = self.model.decode(tensor, self.tokenizer, max_length=max_length)
        pred = preds[0]

        smiles_list, molblock_list, r_success = convert_graph_to_smiles(
            [pred["coords"]], [pred["symbols"]], [pred["edges"]],
            images=[image], num_workers=1)

        out = {
            "smiles": smiles_list[0],
            "raw_smiles": pred["smiles"],
            "molfile": molblock_list[0],
            "graph_success": bool(r_success),
            "num_atoms": len(pred["symbols"]),
            "num_tokens": len(pred["sequence"]),
        }
        if return_graph:
            out["graph"] = {
                "symbols": pred["symbols"],
                "coords": pred["coords"],
                "edges": pred["edges"],
            }
        return out
