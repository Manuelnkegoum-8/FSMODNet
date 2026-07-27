import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from util import MODELS
import math
from torch.autograd import Variable



__all__ = ['QuerySupportAttention', 'CosineLinear', 'Multi_lvl_CosineLinear',
            'PositionEmbeddingSine', 'TaskPositionalEncoding', 'QueryEncoding']

@MODELS.register()
class PositionEmbeddingSine(nn.Module):
    """Sinusoidal position embedding used in DETR model.

    Please see `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for more details.

    Args:
        num_pos_feats (int): The feature dimension for each position along
            x-axis or y-axis. The final returned dimension for each position
            is 2 times of the input value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Default: 10000.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Default: 2*pi.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default: 1e-6.
        offset (float): An offset added to embed when doing normalization.
        normalize (bool, optional): Whether to normalize the position embedding.
            Default: False.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        scale: float = 2 * math.pi,
        eps: float = 1e-6,
        offset: float = 0.0,
        normalize: bool = False,
    ):
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), (
                "when normalize is set,"
                "scale should be provided and in float or int type, "
                f"found {type(scale)}"
            )
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward function for `PositionEmbeddingSine`.

        Args:
            mask (torch.Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for the input tensor. Shape as `(bs, h, w)`.

        Returns:
            torch.Tensor: Returned position embedding with
            shape `(bs, num_pos_feats * 2, h, w)`
        """
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # use view as mmdet instead of flatten for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).view(
            B, H, W, -1
        )
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).view(
            B, H, W, -1
        )
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos





@MODELS.register()
class QuerySupportAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = torch.sqrt(torch.tensor(dim//num_heads, dtype=torch.float32))
        self.q = nn.Linear(dim,dim,bias=False)
        self.linear_out = nn.Sequential(nn.Linear(dim,dim//2),nn.ReLU())
        self.linear_out_2 = nn.Sequential(nn.Linear(dim,dim//2),nn.ReLU())
        self.linear_out_3 = nn.Linear(dim*2,dim)
        self.background = nn.Parameter(torch.randn(1,dim))
        self.d_model = dim
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.normal_(self.q.weight, mean=0, std=np.sqrt(2.0 / (self.d_model + self.d_model)))
        nn.init.xavier_uniform_(self.linear_out[0].weight)
        nn.init.xavier_uniform_(self.linear_out_2[0].weight)
        nn.init.xavier_uniform_(self.linear_out_3.weight)

    def forward(self,query,key,value, pseudo_class=None):
        shortcut = query  
        query = self.q(query)
        key = self.q(key)
        #value = self.v(value)
        bs = query.size(0)
        q = rearrange(query, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(key, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(value, 'b n (h d) -> b h n d', h=self.num_heads)
        background = self.background.unsqueeze(0).expand(q.size(0), -1, -1) # bs 1 d
        background = rearrange(background, 'b n (h d) -> b h n d', h=self.num_heads)
        dummy = torch.zeros_like(background)
        k = torch.cat((k, background),dim=2) # bs h n+1 d
        v = torch.cat((F.sigmoid(v),dummy),dim=2)
        attn = q@k.transpose(-2,-1) # bs h L (n+1)
        attn_weights = F.softmax(attn/self.temperature, dim=-1)
        out = attn_weights@v
        out = out*q
        out = rearrange(out, 'b h n d -> b n (h d)')
        out1 = self.linear_out(out * shortcut)
        out2 = self.linear_out_2(shortcut - out)
        out = torch.cat((out1,out2,shortcut),  dim=-1)
        out = self.linear_out_3(out)
        if pseudo_class is not None:
            #pseudo class  nclass x dim
            pseudo_class = pseudo_class.unsqueeze(0).expand(bs, -1, -1)
            pseudo_class = rearrange(pseudo_class, 'b n (h d) -> b h n d', h=self.num_heads)
            pseudo_class = torch.cat((pseudo_class,dummy),dim=2)
            pseudo_class = attn_weights@pseudo_class
            pseudo_class = rearrange(pseudo_class, 'b h n d -> b n (h d)')
            out = out + pseudo_class
        return out,attn_weights


@MODELS.register()
class TaskPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.05, max_len=10):
        super(TaskPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the task positional encodings once and for all in log space.
        tpe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        tpe[:, 0::2] = torch.sin(position * div_term)
        tpe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('tpe', tpe)

    def forward(self, x):
        x = x + torch.flip(Variable(self.tpe[:x.size(1)], requires_grad=False), [1])
        return self.dropout(x)

@MODELS.register()
class QueryEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=100):
        super(QueryEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the query encodings once and for all in log space.
        queryencoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        queryencoding[:, 0::2] = torch.sin(position * div_term)
        queryencoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('queryencoding', queryencoding)

    def forward(self):
        x = Variable(self.queryencoding, requires_grad=False)
        return self.dropout(x)


class EpisodeSlotEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, slot_ids):
        pe  = torch.flip(Variable(self.pe, requires_grad=False), [1])

        return pe[slot_ids]
    

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, scale=10., bias=False):
        super(CosineLinear, self).__init__()
        self.scale = scale
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1)
        # normalize weight
        self.linear.weight.data = F.normalize(self.linear.weight.data, p=2, dim=-1)
        return self.scale*self.linear(x)

@MODELS.register()
class Multi_lvl_CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, num_scales, scale, bias=False):
        super(Multi_lvl_CosineLinear, self).__init__()
        self.num_feature_levels = num_scales
        self.in_features = in_features
        self.out_features = out_features
        self.layers = nn.ModuleList()
        for i in range(num_scales):
            self.layers.append(CosineLinear(in_features, out_features, scale=scale, bias=bias))
    
    def forward(self, prototypes):
        return torch.stack([self.layers[i](prototypes[0][i].sigmoid()) for i in range(self.num_feature_levels)], dim=0)