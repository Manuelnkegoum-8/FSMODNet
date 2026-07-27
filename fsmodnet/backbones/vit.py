import timm
from util.misc import NestedTensor, nested_tensor_from_tensor_list
import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import Sequence, Tuple, Union, Callable
from util import MODELS

__all__ = ["DINOModel", "TimmDINOModel"]


class ConvNorm(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConvNorm, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # inintialize the weights with normal
        nn.init.kaiming_normal_(self.weight, mode='fan_out')
        if bias:
            nn.init.constant_(self.bias, 0)
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = ConvNorm(in_channels, out_channels, 1, )
        self.norm1 = Norm2d(out_channels)
        self.act = nn.GELU()
        
        
    def forward(self, x):
        conv_out = self.conv1(x)
        conv_out = self.norm1(conv_out)
        conv_out = self.act(conv_out)
        return conv_out
    
@MODELS.register()   
class DINOModel(torch.nn.Module):
    def __init__(self, 
                model, 
                pretrained=None,
                n: Union[int, Sequence] = 1,  # Layers or n last layers to take
                reshape: bool = False,
                return_class_token: bool = False,
                feature_levels: int = 1,
                norm=True,
                **kwargs
                ):
        super().__init__()
        path = os.path.dirname(os.path.abspath(__file__))
        self.model = torch.hub.load(path, model, source="local",pretrained=False, **kwargs)
        if pretrained is not None:
            self.model.load_state_dict(torch.load(pretrained))
        self.embed_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size
        self.n = n
        self.reshape = reshape
        self.return_class_token = return_class_token
        self.feature_levels = feature_levels
        self.norm = norm
        self.dims = [self.embed_dim] * len(self.feature_levels)
        if isinstance(n, int):
           self.in_dims = [self.embed_dim * n] * len(self.feature_levels)
        elif isinstance(n, Sequence):
            self.in_dims = [self.embed_dim * len(n)] * len(self.feature_levels)
        else:
            raise ValueError("n must be an integer or a list of integers")
        
        fpn = []
        if len(self.feature_levels) == 1:
            conv = nn.Identity()
            fpn.append(conv)
            self.dims = self.in_dims
        else:
            for i in self.feature_levels:
                if i == 0: 
                    conv = nn.Sequential(
                                nn.ConvTranspose2d(self.in_dims[0], self.in_dims[0] // 2, kernel_size=2, stride=2),
                                Norm2d(self.in_dims[0] // 2),
                                nn.GELU(),
                                nn.ConvTranspose2d(self.in_dims[0] // 2, self.in_dims[0] // 4, kernel_size=2, stride=2),
                                )
                elif i == 1:
                    conv = nn.Sequential(
                                nn.ConvTranspose2d(self.in_dims[0], self.in_dims[0] // 2, kernel_size=2, stride=2),
                            )
                elif i == 2:
                    conv = nn.Identity()
                else:
                    conv = nn.Sequential(
                                nn.MaxPool2d(2,2**(i-2)),
                            )   
                    if self.in_dims[0] != self.embed_dim:
                        # add a conv layer to match the dimensions
                        conv.insert(0, ConvBlock(self.in_dims[0], self.embed_dim))
                fpn.append(conv)
            
        self.fpn = nn.ModuleList(fpn)
    def forward(self, x):
        B,C,H,W = x.shape
        pad_l = pad_t = 0
        pad_r = (self.patch_size - W % self.patch_size) % self.patch_size
        pad_b = (self.patch_size - H % self.patch_size) % self.patch_size
        # apply bilinear interpolation to pad the image
        x = F.interpolate(x, size=(H + pad_t + pad_b, W + pad_l + pad_r), mode='bilinear', align_corners=False)
        out = self.get_layers(x, self.n, self.reshape, self.return_class_token,self.norm)
        outs = list()
        for idx,i in enumerate(self.feature_levels):
            out_ = self.fpn[idx](out)
            outs.append(out_)
        return outs

    def get_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        
        #h_pad, w_pad = H + pad_t + pad_b, W + pad_l + pad_r
        outputs = self.model._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.model.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        try:
            outputs = [out[:, 1 + self.model.num_register_tokens :] for out in outputs]
        except AttributeError: # dinov3 does not have num_register_tokens attribute
            outputs = [out[:, 1+ self.model.n_storage_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        
        out = torch.cat(outputs,dim=1)
        return out






@MODELS.register()   
class TimmDINOModel(torch.nn.Module):
    def __init__(self, 
                model, 
                pretrained=None,
                n: Union[int, Sequence] = 1,  # Layers or n last layers to take
                reshape: bool = False,
                return_class_token: bool = False,
                feature_levels: int = 1,
                norm=True,
                **kwargs
                ):
        super().__init__()
        self.model = timm.create_model(model, pretrained=True)
        self.embed_dim = self.model.embed_dim
        self.patch_size = self.model.patch_embed.patch_size[0]
        self.n = n
        self.reshape = reshape
        self.return_class_token = return_class_token
        self.feature_levels = feature_levels
        self.norm = norm
        self.dims = [self.embed_dim] * len(self.feature_levels)
        self.in_dims = [self.embed_dim * n] * len(self.feature_levels)
    
        
        fpn = []
        if len(self.feature_levels) == 1:
            conv = nn.Identity()
            fpn.append(conv)
            self.dims = self.in_dims
        else:
            for i in self.feature_levels:
                if i == 0: 
                    conv = nn.Sequential(
                                nn.ConvTranspose2d(self.in_dims[0], self.in_dims[0] // 2, kernel_size=2, stride=2),
                                Norm2d(self.in_dims[0] // 2),
                                nn.GELU(),
                                nn.ConvTranspose2d(self.in_dims[0] // 2, self.in_dims[0] // 4, kernel_size=2, stride=2),
                                )
                elif i == 1:
                    conv = nn.Sequential(
                                nn.ConvTranspose2d(self.in_dims[0], self.in_dims[0] // 2, kernel_size=2, stride=2),
                            )
                elif i == 2:
                    conv = nn.Identity()
                else:
                    conv = nn.Sequential(
                                nn.MaxPool2d(2,2**(i-2)),
                            )   
                    if self.in_dims[0] != self.embed_dim:
                        # add a conv layer to match the dimensions
                        conv.insert(0, ConvBlock(self.in_dims[0], self.embed_dim))
                fpn.append(conv)
            
        self.fpn = nn.ModuleList(fpn)
    def forward(self, x):
        B,C,H,W = x.shape
        pad_l = pad_t = 0
        pad_r = (self.patch_size - W % self.patch_size) % self.patch_size
        pad_b = (self.patch_size - H % self.patch_size) % self.patch_size
        # apply bilinear interpolation to pad the image
        x = F.interpolate(x, size=(H + pad_t + pad_b, W + pad_l + pad_r), mode='bilinear', align_corners=False)
        out = self.model.forward_intermediates(x, self.n,
            return_prefix_tokens=False,
            norm=self.norm,
            output_fmt='NCHW',
            intermediates_only=True,)
        out = torch.cat(out,dim=1)
        outs = list()
        for idx,i in enumerate(self.feature_levels):
            out_ = self.fpn[idx](out)
            outs.append(out_)
        return outs
