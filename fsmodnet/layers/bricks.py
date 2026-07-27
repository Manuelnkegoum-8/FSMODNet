import torch
from torch import nn, Tensor
from util import MODELS
import torch.nn.functional as F

__all__ = ['FFN', 'MLP']

def get_activation(act: str, inpace: bool=True):
    """get activation
    """
    if act is None:
        return nn.Identity()

    elif isinstance(act, nn.Module):
        return act 

    act = act.lower()
    
    if act == 'silu' or act == 'swish':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()

    elif act == 'hardsigmoid':
        m = nn.Hardsigmoid()

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 



def get_norm(cfg: dict):
    """get normalization layer
    """
    if cfg is None:
        return nn.Identity()
    copy_cfg = cfg.copy()
    norm = copy_cfg.pop('type')    
    if norm == 'batchnorm':
        m = nn.BatchNorm2d(**copy_cfg)

    elif norm == 'layernorm':
        m = nn.LayerNorm(**copy_cfg)

    elif norm == 'GN':
        m = nn.GroupNorm(**copy_cfg)

    else:
        raise RuntimeError('')  

    return m

@MODELS.register()
class FFN(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        return src


@MODELS.register()
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
