from transformers import AutoConfig, AutoModel

from .configuration_swin_ocsr import SwinOCSRConfig
from .modeling_swin_ocsr import SwinOCSRModel

AutoConfig.register("swin_ocsr", SwinOCSRConfig)
AutoModel.register(SwinOCSRConfig, SwinOCSRModel)
