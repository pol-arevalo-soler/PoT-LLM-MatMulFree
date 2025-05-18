import torch

def activation_quant(x):
    """
    Per-token quantization to 8 bits. No grouping is needed for quantization.

    Args:
        x: An activation tensor with shape [n, d].

    Returns:
        A quantized activation tensor with shape [n, d].
    """
    
    
    # Compute the scale factor
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    # Quantize and then de-quantize the tensor
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y
    


    # Downstream 11: no Quantization
    #return x
    
def weight_quant(w):
    """
    Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.

    Args:
        w: A weight tensor with shape [d, k].

    Returns:
        A quantized weight tensor with shape [d, k].
    """
    
    # Threshold  (always keep this not commented)
    threshold = 0.5

    '''
    # Downstream 1: int1.58
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    # Quantize and then de-quantize the tensor
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u
    '''

    '''
    # Downstream 2: int4 (column-wise)
    scale = 7.0 / w.abs().max(dim=0, keepdim=True).values.clamp_(min=1e-5)      
    u = (w * scale).round().clamp_(-8, 7) / scale
    return u
    '''

    '''
    # Downstream 3: int4 max of all matrix
    scale = 7.0 / w.abs().max().clamp_(min=1e-5)      
    u = (w * scale).round().clamp_(-8, 7) / scale
    return u
    '''

    '''
    # Downstream 4: Powers of Two 4-bit column-wise
    scale = 64.0 / w.abs().max(dim=0, keepdim=True).values.clamp_(min=1e-5)
    quant = power_of_2(scale * w, threshold)
    return (quant / scale)
    '''

    '''
    # Downstream 5: Powers of Two 4-bit sergment-wise (block-size = 8)
    n, m = w.shape
    block_size = 8

    if n % block_size != 0:
        scale = 64.0 / w.abs().max(dim=0, keepdim=True).values.clamp_(min=1e-5)
        quant = power_of_2(scale * w, threshold)
        return (quant / scale)
    
    w_reshaped = w.view(n // block_size, block_size, m)

    # Compute the absolute max per column for each block (along rows)
    scale = 64.0 / w_reshaped.abs().max(dim=1, keepdim=True).values.clamp_(min=1e-5)
    # Multiply element-wise by the max and reshape back to the original form
    quant = power_of_2((scale * w_reshaped), threshold) / scale

    return quant.view(n, m)
    '''

    '''
    # Downstream 6: segment-wise with max
    n, m = w.shape
    block_size = 8

    if n % block_size != 0:
        scale = 7.0 / w.abs().max(dim=0, keepdim=True).values.clamp_(min=1e-5)      
        u = (w * scale).round().clamp_(-8, 7) / scale
        return u
    
    w_reshaped = w.view(n // block_size, block_size, m)

    # Compute the absolute max per column for each block (along rows)
    scale = 7.0 / w_reshaped.abs().max(dim=1, keepdim=True).values.clamp_(min=1e-5)
    # Multiply element-wise by the max and reshape back to the original form
    u = (scale * w_reshaped).round().clamp_(-8, 7) / scale

    return u.view(n, m)
    '''

    
    # Downstream 7: PoT segment-wise block_size = 2
    n, m = w.shape
    block_size = 2

    if n % block_size != 0:
        scale = 64.0 / w.abs().max(dim=0, keepdim=True).values.clamp_(min=1e-5)
        quant = power_of_2(scale * w, threshold)
        return (quant / scale)
    
    w_reshaped = w.view(n // block_size, block_size, m)

    # Compute the absolute max per column for each block (along rows)
    scale = 64.0 / w_reshaped.abs().max(dim=1, keepdim=True).values.clamp_(min=1e-5)
    # Multiply element-wise by the max and reshape back to the original form
    quant = power_of_2((scale * w_reshaped), threshold) / scale

    return quant.view(n, m)
    
    
    '''
    # Downstream 8: PoT segment-wise block_size = 4
    n, m = w.shape
    block_size = 4

    if n % block_size != 0:
        scale = 64.0 / w.abs().max(dim=0, keepdim=True).values.clamp_(min=1e-5)
        quant = power_of_2(scale * w, threshold)
        return (quant / scale)
    
    w_reshaped = w.view(n // block_size, block_size, m)

    # Compute the absolute max per column for each block (along rows)
    scale = 64.0 / w_reshaped.abs().max(dim=1, keepdim=True).values.clamp_(min=1e-5)
    # Multiply element-wise by the max and reshape back to the original form
    quant = power_of_2((scale * w_reshaped), threshold) / scale

    return quant.view(n, m)
    '''

    '''
    # Downstream 9: segment-wise (block_size = 2) with max
    n, m = w.shape
    block_size = 2

    if n % block_size != 0:
        scale = 7.0 / w.abs().max(dim=0, keepdim=True).values.clamp_(min=1e-5)      
        u = (w * scale).round().clamp_(-8, 7) / scale
        return u
    
    w_reshaped = w.view(n // block_size, block_size, m)

    # Compute the absolute max per column for each block (along rows)
    scale = 7.0 / w_reshaped.abs().max(dim=1, keepdim=True).values.clamp_(min=1e-5)
    # Multiply element-wise by the max and reshape back to the original form
    u = (scale * w_reshaped).round().clamp_(-8, 7) / scale

    return u.view(n, m)
    '''

    '''
    # Downstream 10: segment-wise (block_size = 4) with max
    n, m = w.shape
    block_size = 4

    if n % block_size != 0:
        scale = 7.0 / w.abs().max(dim=0, keepdim=True).values.clamp_(min=1e-5)      
        u = (w * scale).round().clamp_(-8, 7) / scale
        return u
    
    w_reshaped = w.view(n // block_size, block_size, m)

    # Compute the absolute max per column for each block (along rows)
    scale = 7.0 / w_reshaped.abs().max(dim=1, keepdim=True).values.clamp_(min=1e-5)
    # Multiply element-wise by the max and reshape back to the original form
    u = (scale * w_reshaped).round().clamp_(-8, 7) / scale

    return u.view(n, m)
    '''

    
    # Downstream 11: no quantization
    #return w

def nearest_power(w : torch.Tensor, threshold : float) -> torch.Tensor:
             
    # Nearest power of 2 of scaled tensor
    exponent = torch.log2((w).clamp_(min=2**-8))
    quant = torch.exp2((exponent - 0.084).round()).clamp(min=1.0, max=64.0) # 4 bits: min=2^-7, max=2^0

    return torch.where(w < threshold, torch.zeros_like(w), quant)

def power_of_2(w : torch.Tensor, threshold : float) -> torch.Tensor:
    # Quantization 
    quant = nearest_power(torch.abs(w), threshold)
    # Sign
    sign = torch.sign(w)
    quant *= sign 

    return quant