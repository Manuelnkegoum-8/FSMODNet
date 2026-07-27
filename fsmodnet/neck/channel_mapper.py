import torch
import torch.nn as nn
from util import MODELS
from ..layers.bricks import get_activation, get_norm




__all__ = ['ChannelMapper']



class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 bias=False, groups=1, dilation=1, padding=0,
                 norm_cfg=None, act_cfg=None, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              bias=bias, groups=groups, dilation=dilation, padding=padding, **kwargs)
        self.norm = get_norm(norm_cfg) if norm_cfg is not None else None
        self.act = get_activation(act_cfg) if act_cfg is not None else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x

@MODELS.register()
class ChannelMapper(nn.Module):
    def __init__(self, 
                 in_channels=[512, 1024, 20148], 
                 kernel_size=1, stride=1,
                 bias=False, groups=1, dilation=1,
                 norm_cfg=None, act_cfg=None,
                 out_channels=256, num_outs=4, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        
        self.convs = nn.ModuleList()
        self.extra_convs = nn.ModuleList()
        for in_channel in in_channels:
            conv = ConvNormAct(in_channel, out_channels, kernel_size=kernel_size, stride=stride,
                               bias=bias, groups=groups, dilation=dilation, padding=(kernel_size-1)//2,
                               norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.convs.append(conv)

        if num_outs > len(in_channels):
            for i in range(len(in_channels), num_outs):
                if i == len(in_channels):
                    in_c = in_channels[-1]
                else:
                    in_c = out_channels
                conv = ConvNormAct(in_c, out_channels, kernel_size=3, bias=bias, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
                self.extra_convs.append(conv)


    def forward(self, x):
        assert len(x) == len(self.in_channels)
        outs = []
        for i, conv in enumerate(self.convs):
            outs.append(conv(x[i]))

        if len(self.extra_convs) > 0:
            for i, conv in enumerate(self.extra_convs):
                if i == 0:
                    outs.append(conv(x[-1]))
                else:
                    outs.append(conv(outs[-1]))

        return tuple(outs)


