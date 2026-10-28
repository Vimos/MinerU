# coding=utf-8
"""SwinOCSR model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SwinOCSRConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SwinOCSRForConditionalGeneration`]. It is used to instantiate an
    SwinOCSR model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the SwinOCSR-9B.

    e.g. [llava-hf/llava-9b](https://huggingface.co/llava-hf/llava-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_layer (int): the number of GNN layers
        arg.emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536

    Example:

    ```python
    >>> from transformers import LlamaConfig
    >>> from reactigraph.models.molecule_stm.configuration_molecule_stm import SwinOCSRConfig
    >>> from reactigraph.models.molecule_stm.modeling_molecule_stm import SwinOCSRForGraphPrediction

    >>> # Initializing a molecule config
    >>> molecule_config = SwinOCSRConfig()

    >>> # Initializing a model from the llava-1.5-7b style configuration
    >>> model = SwinOCSRForGraphPrediction(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "swin_ocsr"
    is_composition = True

    def __init__(
            self,
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=192,
            depths=None,
            num_heads=None,
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            ape=False,
            patch_norm=True,
            dim=256,
            initializer_range: float = 0.02,
            **kwargs,
    ):
        if num_heads is None:
            num_heads = [6, 12, 24, 48]
        if depths is None:
            depths = [2, 2, 18, 2]
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.ape = ape
        self.patch_norm = patch_norm
        self.dim = dim
        self.attn_drop_rate = attn_drop_rate
        self.initializer_range = initializer_range

        self.decoder_dim = 256  # dimension of decoder RNN
        self.ff_dim = 2048
        self.num_head = 8
        self.dropout = 0.1
        self.encoder_num_layer = 6
        self.decoder_num_layer = 6
        self.max_len = 277
        self.vocab_size = 79

        super().__init__(**kwargs)


__all__ = ["SwinOCSRConfig"]
