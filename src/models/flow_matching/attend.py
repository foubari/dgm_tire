"""
Attention mechanism with flash attention support
"""

from functools import wraps
from packaging import version
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange

AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

class Attend(nn.Module):
    def __init__(self, dropout=0., flash=False, scale=None):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)
        
        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), \
            'in order to use flash attention, you must be using pytorch 2.0 or above'
        
        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None
        
        if not torch.cuda.is_available() or not flash:
            return
        
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        device_name = device_properties.name
        device_version = version.parse(f'{device_properties.major}.{device_properties.minor}')
        
        is_a100 = 'A100' in device_name.upper() or 'A100' in device_name
        
        if is_a100:
            print_once(f'A100 GPU detected ({device_name}), using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(True, False, False)
        elif device_version >= version.parse('8.0'):
            print_once(f'Modern GPU detected ({device_name}, compute {device_properties.major}.{device_properties.minor}), using flash attention with mem_efficient fallback')
            self.cuda_config = AttentionConfig(True, False, True)
        else:
            print_once(f'Non-A100 GPU detected ({device_name}), using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(False, True, True)
    
    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device
        
        if exists(self.scale):
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)
        
        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
        
        config = self.cuda_config if is_cuda else self.cpu_config
        
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.
            )
        
        return out
    
    def forward(self, q, k, v):
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device
        
        if self.flash:
            return self.flash_attn(q, k, v)
        
        scale = default(self.scale, q.shape[-1] ** -0.5)
        
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)
        
        return out

