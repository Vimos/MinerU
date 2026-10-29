"""MolGrapher backend (PatCID recognition model, ICCV'23).

Pure-torch re-implementation: resnet18 keypoint detector + resnet50/GCN
graph classifier, loading the official ``ds4sd/MolGrapher`` checkpoints.
"""
from .interface import VARIANTS, MolGrapher

__all__ = ["MolGrapher", "VARIANTS"]
