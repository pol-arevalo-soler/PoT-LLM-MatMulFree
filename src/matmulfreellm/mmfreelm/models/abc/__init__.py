# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from mmfreelm.models.abc.configuration_abc import ABCConfig
from mmfreelm.models.abc.modeling_abc import ABCForCausalLM, ABCModel

AutoConfig.register(ABCConfig.model_type, ABCConfig)
AutoModel.register(ABCConfig, ABCModel)
AutoModelForCausalLM.register(ABCConfig, ABCForCausalLM)


__all__ = ['ABCConfig', 'ABCForCausalLM', 'ABCModel']
