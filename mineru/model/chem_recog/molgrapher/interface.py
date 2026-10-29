"""MolGrapher facade: checkpoints + image -> SMILES.

    from ocsr import load_ocsr
    m = load_ocsr("molgrapher")
    m.predict_image(rgb_uint8)["smiles"]

Pure torch: no pytorch_lightning, no torch_geometric, no mol_depict, no cv2,
no mahotas, no PaddleOCR.
"""
import os

import numpy as np
import torch

from ..base import OCSRBase
from .graph_classifier import GraphClassifier
from .graph_constructor import GraphConstructor
from .keypoint_detector import KeypointDetector
from .molecular_graph import MolecularGraph
from .preprocess import preprocess_image, to_tensor

__all__ = ["MolGrapher", "VARIANTS"]

CKPT_REPO = "ds4sd/MolGrapher"
KD_FILE = "models/keypoint_detector/kd_model.ckpt"

#: variant -> (checkpoint file, nb_atoms_classes, nb_bonds_classes, gcn_on)
VARIANTS = {
    "gc_gcn_model": ("models/graph_classifier/gc_gcn_model.ckpt", 141, 5, True),
    "gc_no_stereo_model": ("models/graph_classifier/gc_no_stereo_model.ckpt", 182, 6, False),
    "gc_stereo_model": ("models/graph_classifier/gc_stereo_model.ckpt", 182, 8, False),
}

_HERE = os.path.dirname(os.path.abspath(__file__))


def download_checkpoint(filename, repo=CKPT_REPO, local_dir=None):
    if local_dir:
        local = os.path.join(local_dir, filename)
        if os.path.exists(local):
            return local
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename)


def _load_state_dict(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        hparams = ckpt.get("hyper_parameters", {})
    else:  # plain state dict
        state = ckpt
        hparams = {}
    # lightning "model." prefix is not used by the released ckpts, but tolerate it
    if all(k.startswith("model.") for k in state):
        state = {k[len("model."):]: v for k, v in state.items()}
    return state, hparams


class MolGrapher(OCSRBase):
    """PatCID / MolGrapher (ICCV'23) recognition model, pure-torch port."""

    name = "molgrapher"

    def __init__(self, device=None, variant="gc_gcn_model", keypoint_checkpoint=None,
                 graph_checkpoint=None, image_size=1024, border_size=30,
                 local_dir=None, verbose=False):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {list(VARIANTS)}")
        self.variant = variant
        gc_file, nb_atoms, nb_bonds, gcn_on = VARIANTS[variant]
        self.nb_atoms_classes = nb_atoms
        self.nb_bonds_classes = nb_bonds
        self.gcn_on = gcn_on

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.border_size = border_size
        self.verbose = verbose
        self.config_dataset_keypoint = {"image_size": [3, image_size, image_size],
                                        "mask_size": [1, image_size // 4,
                                                      image_size // 4]}
        self.config_dataset_graph = {"image_size": [3, image_size, image_size],
                                     "nb_atoms_classes": nb_atoms,
                                     "nb_bonds_classes": nb_bonds}

        keypoint_checkpoint = keypoint_checkpoint or download_checkpoint(
            KD_FILE, local_dir=local_dir)
        graph_checkpoint = graph_checkpoint or download_checkpoint(
            gc_file, local_dir=local_dir)
        self.keypoint_checkpoint = keypoint_checkpoint
        self.graph_checkpoint = graph_checkpoint

        # ---- keypoint detector ---- #
        self.keypoint_detector = KeypointDetector()
        kd_state, kd_hparams = _load_state_dict(keypoint_checkpoint)
        self.kd_report = self._strict_load(self.keypoint_detector, kd_state,
                                           "keypoint_detector")
        self.kd_hparams = kd_hparams

        # ---- graph classifier ---- #
        self.graph_classifier = GraphClassifier(nb_atoms_classes=nb_atoms,
                                                nb_bonds_classes=nb_bonds,
                                                gcn_on=gcn_on)
        gc_state, gc_hparams = _load_state_dict(graph_checkpoint)
        self.gc_report = self._strict_load(self.graph_classifier, gc_state,
                                           "graph_classifier")
        self.gc_hparams = gc_hparams

        self.keypoint_detector.to(self.device).eval()
        self.graph_classifier.to(self.device).eval()
        self._load_report = {"keypoint_detector": self.kd_report,
                             "graph_classifier": self.gc_report}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _strict_load(module, state, name):
        own = module.state_dict()
        missing = [k for k in own if k not in state]
        unexpected = [k for k in state if k not in own]
        shape_mismatch = {k: (tuple(state[k].shape), tuple(own[k].shape))
                          for k in own if k in state
                          and tuple(state[k].shape) != tuple(own[k].shape)}
        result = module.load_state_dict(state, strict=False)
        report = {"missing": missing or list(result.missing_keys),
                  "unexpected": unexpected or list(result.unexpected_keys),
                  "shape_mismatch": shape_mismatch,
                  "n_keys_ckpt": len(state), "n_keys_module": len(own),
                  "strict_ok": (not missing and not unexpected
                                and not shape_mismatch)}
        if report["strict_ok"]:
            module.load_state_dict(state, strict=True)  # proof
        else:
            print(f"[{name}] MISSING {missing}\n[{name}] UNEXPECTED {unexpected}\n"
                  f"[{name}] SHAPE {shape_mismatch}")
        return report

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict_image(self, image, return_all=False, verbose=None):
        verbose = self.verbose if verbose is None else verbose
        rgb = self.to_numpy_rgb(image)
        pil = preprocess_image(rgb, self.image_size, self.border_size)
        tensor = to_tensor(pil).to(self.device)

        # 1. keypoints
        kp = self.keypoint_detector.predict(tensor, verbose=verbose)
        keypoints = kp["keypoints_batch"][0]

        out = {"smiles": None, "confidence": 0.0, "keypoints": keypoints,
               "num_keypoints": len(keypoints), "num_atoms": 0, "num_bonds": 0,
               "molfile": None, "graph_success": False, "variant": self.variant,
               "abbreviations": [], "unresolved_abbreviations": []}
        if len(keypoints) < 2:
            out["error"] = "not enough keypoints"
            if return_all:
                out["heatmap"] = kp["heatmaps_batch"][0]
            return out

        # 2. dense graph structure
        constructor = GraphConstructor(keypoints, tensor[0],
                                       self.config_dataset_keypoint,
                                       self.config_dataset_graph,
                                       discarded_bond_length_factor=1.75)
        structure = constructor.build_structure()

        # 3. node / edge classification
        atoms_classes, bonds_classes, confidences = self.graph_classifier.predict(
            tensor, [structure])

        # 4. RDKit molecule
        graph = MolecularGraph(self.config_dataset_graph)
        graph.from_predictions_nodes_only(atoms_classes[0], bonds_classes[0], structure)
        smiles, molblock = graph.to_smiles()

        out.update({
            "smiles": smiles,
            "confidence": 0.0 if graph.fallback else confidences[0],
            "num_atoms": len(graph.atoms),
            "num_bonds": len(graph.bonds),
            "molfile": molblock,
            "graph_success": smiles is not None and not graph.fallback,
            "abbreviations": graph.abbreviations_used,
            "unresolved_abbreviations": graph.unresolved_abbreviations,
            "n_dropped_atoms": graph.n_dropped_atoms,
            "n_dropped_bonds": graph.n_dropped_bonds,
        })
        # the molecule is only kekulised (upstream behaviour), so validate the
        # emitted SMILES and give consumers a canonical form as well
        try:
            from ..common.chemistry import canonical_smiles, mol_from_smiles

            out["valid_smiles"] = mol_from_smiles(smiles) is not None
            out["canonical_smiles"] = canonical_smiles(smiles)
        except Exception:
            out["valid_smiles"] = None
            out["canonical_smiles"] = None
        if return_all:
            out["structure"] = structure
            out["heatmap"] = kp["heatmaps_batch"][0]
            out["preprocessed"] = np.asarray(pil)
            out["atoms_classes"] = atoms_classes[0]
            out["bonds_classes"] = bonds_classes[0]
        if verbose:
            print(f"keypoints={len(keypoints)} atoms={out['num_atoms']} "
                  f"bonds={out['num_bonds']} smiles={smiles}")
        return out
