# -*- coding: utf-8 -*-

from mmfreelm.modules.convolution import ImplicitLongConvolution, LongConvolution, ShortConvolution
from mmfreelm.modules.fused_bitlinear import BitLinear, FusedBitLinear
from mmfreelm.modules.fused_cross_entropy import FusedCrossEntropyLoss
from mmfreelm.modules.fused_kl_div import FusedKLDivLoss
from mmfreelm.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from mmfreelm.modules.fused_norm_gate import (
    FusedLayerNormGated,
    FusedLayerNormSwishGate,
    FusedLayerNormSwishGateLinear,
    FusedRMSNormGated,
    FusedRMSNormSwishGate,
    FusedRMSNormSwishGateLinear
)
from mmfreelm.modules.layernorm import GroupNorm, GroupNormLinear, LayerNorm, LayerNormLinear, RMSNorm, RMSNormLinear
from mmfreelm.modules.mlp import GatedMLP
from mmfreelm.modules.mlp_bit import GatedMLPBit
from mmfreelm.modules.rotary import RotaryEmbedding

from mmfreelm.modules.quantization import activation_quant, weight_quant

__all__ = [
    'ImplicitLongConvolution', 'LongConvolution', 'ShortConvolution',
    'BitLinear', 'FusedBitLinear',
    'FusedCrossEntropyLoss', 'FusedLinearCrossEntropyLoss', 'FusedKLDivLoss',
    'GroupNorm', 'GroupNormLinear', 'LayerNorm', 'LayerNormLinear', 'RMSNorm', 'RMSNormLinear',
    'FusedLayerNormGated', 'FusedLayerNormSwishGate', 'FusedLayerNormSwishGateLinear',
    'FusedRMSNormGated', 'FusedRMSNormSwishGate', 'FusedRMSNormSwishGateLinear',
    'GatedMLP', 'GatedMLPBit',
    'RotaryEmbedding',
    'activation_quant', 'weight_quant'
]
