# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from mmfreelm.models.rwkv6.configuration_rwkv6 import RWKV6Config
from mmfreelm.models.rwkv6.modeling_rwkv6 import RWKV6ForCausalLM, RWKV6Model

AutoConfig.register(RWKV6Config.model_type, RWKV6Config, True)
AutoModel.register(RWKV6Config, RWKV6Model, True)
AutoModelForCausalLM.register(RWKV6Config, RWKV6ForCausalLM, True)


__all__ = ['RWKV6Config', 'RWKV6ForCausalLM', 'RWKV6Model']
