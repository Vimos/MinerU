"""MolScribe backend (clean, timm/OpenNMT/albumentations-free).

    from ocsr import load_ocsr
    m = load_ocsr("molscribe")
    m.predict_image(rgb_uint8_array)["smiles"]

The class keeps the exact API of the verified ``ms_clean`` package it was
migrated from (``MolScribe(model_path, vocab_path, device)`` with
``predict_image(image, return_graph=False, max_length=480)``); the only change
is that ``vocab_path``/``model_path`` may now be omitted, in which case the
packaged vocabulary / the ``yujieq/MolScribe`` HF checkpoint are used.
"""
import os

from .interface import MolScribe
from .model import MolScribeNet, Encoder, Decoder
from .tokenizer import CharTokenizer
from .preprocess import preprocess_image

_HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB = os.path.join(_HERE, "vocab", "vocab_chars.json")

__all__ = ["MolScribe", "MolScribeNet", "Encoder", "Decoder", "CharTokenizer",
           "preprocess_image", "VOCAB"]
