import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from util import MODELS
from natten.functional import na2d

__all__ = ['SpectralAttentionFusionNeck', 'SpectralAttentionv3', 'SpectralAttentionv4']


class NeighborhoodAttention(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=5, dilation=1, attn_drop=0.1, proj_drop=0.1):
        super(NeighborhoodAttention, self).__init__()
        self.fp16_enabled = False
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [3, 5, 7, 9, 11, 13], \
            f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size
        assert dilation is None or dilation >= 1, \
                f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.q = nn.Conv2d(dim, dim, 1, padding=0)
        self.kv = nn.Conv2d(dim, dim * 2, 1, padding=0)
        #self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1)))
        #nn.init.trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, C, H, W = x.shape
        # pad x and y to multiples of window size
        pad_l = pad_t = pad_r = pad_b = 0
        if H < self.window_size or W < self.window_size:
            pad_l = max(self.window_size - W, 0)
            pad_b = max(self.window_size - H, 0)
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            y = F.pad(y, (pad_l, pad_r, pad_t, pad_b))
            H_, W_ = x.shape[2], x.shape[3]

        q = self.q(x).reshape(B,self.num_heads, C // self.num_heads, H , W).permute(0,3,4,1,2) # B d H W h
        kv = self.kv(y).reshape(B, 2, self.num_heads, C // self.num_heads, H ,  W).permute(1,0,4,5,2,3) # 3 B h H W d
        k, v = kv[0], kv[1] # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        #attn = na2d(query=q, key=k, rpb=self.rpb, kernel_size=self.kernel_size, dilation=self.dilation, is_causal=False)
        x = na2d(query=q, key=k, value=v, kernel_size=self.kernel_size, dilation=self.dilation, is_causal=False)
        """attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = na2d_av(attn=attn, value=v, kernel_size=self.kernel_size, dilation=self.dilation, is_causal=False)"""
        x = x.permute(0, 3, 4, 1, 2).reshape(B, C, H, W) # B d h w c
        if pad_r or pad_b:
            x = x[:, :, :H_, :W_]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class FeedForward(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LPU(nn.Module):
    def __init__(self, dim):
        super(LPU, self).__init__()
        self.dim = dim
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
    
    def forward(self, x):
        x1 = x
        x = self.dwc(x)
        x = x1 + x
        return x

class ConvFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop=0.):
        super(ConvFFN, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim*mlp_ratio, 1, padding=0)
        self.conv2 = nn.Conv2d(dim*mlp_ratio, dim, 1, padding=0)
        self.dwc = nn.Conv2d(dim*mlp_ratio, dim*mlp_ratio, 3, padding=1, groups=dim*mlp_ratio)
        self.drop = nn.Dropout(drop)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.drop(x)
        x =  x + self.dwc(x)
        x = F.gelu(x)
        x = self.drop(self.conv2(x))
        return x

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm that works on 4D input (N, C, H, W)"""

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__(normalized_shape, eps, elementwise_affine)
    
    def forward(self, x):
        # Reshape x to (N, C, H*W) for LayerNorm
        N, C, H, W = x.size()
        x = x.view(N, C, -1).contiguous()
        x = x.permute(0, 2, 1).contiguous()  # (N, H*W, C)
        # Apply LayerNorm
        x = super().forward(x)
        x = x.permute(0, 2, 1).contiguous()  # (N, C, H*W)
        # Reshape back to (N, C, H, W)
        return x.view(N, C, H, W).contiguous()
    



class CrossDeformableAttention(nn.Module):
    def __init__(self, dim, num_groups, num_heads, ks = 5 , stride=4, attn_drop=0.1, scale=2.0, size= 80, **kwargs):
        super(CrossDeformableAttention, self).__init__()
        assert dim % num_groups == 0, 'dim must be divisible by num_groups'
        assert num_heads % num_groups == 0, 'num_heads must be divisible by num_groups'
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.scale = scale
        self.stride = stride
        self.group_dim = dim//num_groups
        self.temperature = torch.sqrt(torch.tensor(self.group_dim//num_heads, dtype=torch.float32))
        pad = 0 if stride==ks else ks//2
        self.offset_network = nn.Sequential(
                                        nn.Conv2d(self.group_dim, self.group_dim, ks, padding=pad , stride=stride, groups=self.group_dim),
                                        LayerNorm2d(self.group_dim),
                                        nn.GELU(),
                                        nn.Conv2d(self.group_dim,2,1,padding=0,bias=False),
                                        )
        self.q = nn.Conv2d(dim,dim,1,padding=0)
        self.k = nn.Conv2d(dim,dim,1,padding=0)
        self.v = nn.Conv2d(dim,dim,1,padding=0)
        self.o = nn.Conv2d(dim,dim,1,padding=0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.drop = nn.Dropout(attn_drop)
        self.size = size
        if isinstance(size, int):
            self.rpe_table = nn.Parameter(
                    torch.zeros(num_heads, size * 2 - 1, size * 2 - 1)
                )
        elif isinstance(size, tuple) and len(size) == 2:
            self.rpe_table = nn.Parameter(
                        torch.zeros(self.num_heads, self.size[0] * 2 - 1, self.size[1] * 2 - 1)
                    )
        else:
            raise ValueError("size must be an int or a tuple of two ints")
        trunc_normal_(self.rpe_table, std=0.01)
        self._reset_parameters()
    

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
        #nn.init.xavier_uniform_(self.proj_k.weight)
        nn.init.constant_(self.o.bias, 0)   
         
    @torch.no_grad()
    def _get_reference_points(self, H, W, dtype):
        ref_points = torch.stack(torch.meshgrid(
                                                torch.linspace(0.5, H-0.5, H, dtype=dtype), 
                                                torch.linspace(0.5, W-0.5, W, dtype=dtype)
                                                ), dim=-1)
        ref_points[:,:,0].mul_(2).div_(H-1).sub_(1)
        ref_points[:,:,1].mul_(2).div_(W-1).sub_(1)
        return ref_points
    
    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.num_groups, -1, -1, -1) # B * g H W 2

        return ref
    
    def forward(self, query,key):
        B, C, H, W = query.size()
        dtype, device = query.dtype, query.device
        # get offsets from the key image
        proj_q = self.q(query)
        proj_k = self.k(key)
        proj_k = rearrange(proj_k, 'b (g d) h w -> (b g) d h w ', g=self.num_groups) # bg d h w
        offsets = self.offset_network(proj_k).contiguous().permute(0,2,3,1) # bg hr wr 2
        hr,wr = offsets.size(1),offsets.size(2)
        n_sample = hr * wr
        # get reference points
        ref_points = self._get_reference_points(hr,wr,query.dtype).to(device)
        ref_points = ref_points.unsqueeze(0).expand_as(offsets) #bg hr wr 2
        offset_range = torch.tensor([1.0 / (hr - 1.0), 1.0 / (wr - 1.0)], device=query.device, dtype=query.dtype).reshape(1, 1, 1,2)
        pos = offset_range*self.scale*F.tanh(offsets)+ref_points # bg hr wr 2
        #offsets = offsets+ref_points
        #pos = offsets.clamp(-1., +1.)

        # get pixels via bilinear interpolation
        key = rearrange(key, 'b (g d) h w -> (b g) d h w ', g=self.num_groups) # bg d h w
        sampled = F.grid_sample(key,pos[..., (1,0)],mode='bilinear',align_corners=True)
        sampled = rearrange(sampled, '(b g) d hr wr -> b (g d) hr wr ', g=self.num_groups) # b gd hr wr

        # get query, key and value
        k = self.k(sampled) # b c hr wr
        v = self.v(sampled) # b c hr wr
        q = rearrange(proj_q, 'b (i d) h w -> b i (h w) d ', i=self.num_heads) # b i hw d
        k = rearrange(k, 'b (i d) h w -> b i (h w) d ', i=self.num_heads) # b i hw d
        v = rearrange(v, 'b (i d) h w -> b i (h w) d ', i=self.num_heads) #b i hw d
        # compute attention
        attn = q@k.transpose(-2,-1)
        attn = attn/self.temperature
        
        rpe_table = self.rpe_table
        rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
        rpe_bias =  F.interpolate(rpe_bias, size=(2*H - 1, 2*W - 1), mode="bilinear", align_corners=False)
        
        q_grid = self._get_q_grid(H, W, B, dtype, device)
        displacement = (q_grid.reshape(B * self.num_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.num_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
        attn_bias = F.grid_sample(
                    input=rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.num_heads // self.num_groups, g=self.num_groups),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True) # B * g, h_g, HW, Ns

        attn_bias = attn_bias.reshape(B,self.num_groups, self.num_heads // self.num_groups, H * W, n_sample)
        attn_bias = attn_bias.view(B,-1, H * W, n_sample) # B g h_g hw n_sample
        attn = attn + attn_bias
        
        attn = F.softmax(attn, dim=-1)
        out = self.attn_drop(attn)@v
        out = rearrange(out, 'b i (h w) d -> b (i d) h w ', h=H, w=W) 
        out = self.drop(self.o(out)) 
        return out
    
class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, groups=4, stride=4, mlp_ratio=2,attn_type = 'L',scale=2., nat_ks=3, kernel_size=5, 
                        dilation=1, attn_drop=0.1, proj_drop=0.1, size=80, **kwargs):
        super(AttentionBlock, self).__init__()
        self.norm1 = LayerNorm2d(dim)
        self.norm1_ = LayerNorm2d(dim)

        if attn_type == 'L':
            self.attn = NeighborhoodAttention(dim, num_heads, nat_ks, dilation, attn_drop, proj_drop)
            self.attn2 = NeighborhoodAttention(dim, num_heads, nat_ks, dilation, attn_drop, proj_drop)
        elif attn_type == 'D':
            self.attn = CrossDeformableAttention(dim,groups,num_heads,kernel_size,stride,attn_drop,scale,size)
            self.attn2 = CrossDeformableAttention(dim,groups,num_heads,kernel_size,stride,attn_drop,scale,size)

        else:
            raise ValueError(f"Unsupported attention type: {attn_type}")
        self.norm2 = LayerNorm2d(dim)
        self.norm2_ = LayerNorm2d(dim)
        self.mlp = ConvFFN(dim, mlp_ratio=mlp_ratio, drop=0.1)
        self.mlp2 = ConvFFN(dim, mlp_ratio=mlp_ratio, drop=0.1)

        """self.lpu = LPU(dim)
        self.lpu2 = LPU(dim)"""
        self.drop_path = DropPath(0.1) if 0.1 > 0. else nn.Identity()
        self.layer_scale1 = nn.Identity()
        self.layer_scale2 = nn.Identity()
        self.layer_scale3 = nn.Identity()
        self.layer_scale4 = nn.Identity()

    def forward(self, x, y):
        # x B C H W
        
        input1 = x#self.lpu(x)
        input2 = y#self.lpu2(y)
        
        shortcut = input1
        shortcut2 = input2

        input1 = self.norm1(input1)
        input2 = self.norm1_(input2)
        input1_ = self.attn(input1, input2)
        input2_ = self.attn2(input2, input1)

        input1 = self.drop_path(self.layer_scale1(input1_)) + shortcut
        input2 = self.drop_path(self.layer_scale2(input2_))  + shortcut2

        x_ = self.norm2(input1)
        y_ = self.norm2_(input2)

        input1 = self.drop_path(self.layer_scale3(self.mlp(x_))) + input1
        input2 = self.drop_path(self.layer_scale4(self.mlp2(y_))) + input2

        return input1, input2
    

class AttentionBlock2(nn.Module):
    def __init__(self, dim, num_heads, groups=4, stride=4, mlp_ratio=2,scale=2., nat_ks=3,
                  kernel_size=5, dilation=1, attn_drop=0.1, proj_drop=0.1, size=80, **kwargs):
        super(AttentionBlock2, self).__init__()
        self.norm1 = LayerNorm2d(dim)
        self.norm1_ = LayerNorm2d(dim)

        self.attn = NeighborhoodAttention(dim, num_heads, nat_ks, dilation, attn_drop, proj_drop)
        self.attn2 = NeighborhoodAttention(dim, num_heads, nat_ks, dilation, attn_drop, proj_drop)
        
        self.cross_attn = CrossDeformableAttention(dim,groups,num_heads,kernel_size,stride,attn_drop,scale,size)
        self.cross_attn2 = CrossDeformableAttention(dim,groups,num_heads,kernel_size,stride,attn_drop,scale,size)

        self.norm2 = LayerNorm2d(dim)
        self.norm2_ = LayerNorm2d(dim)
        self.mlp = ConvFFN(dim, mlp_ratio=mlp_ratio, drop=proj_drop)
        self.mlp2 = ConvFFN(dim, mlp_ratio=mlp_ratio, drop=proj_drop)
        self.norm3 = LayerNorm2d(dim)
        self.norm3_ = LayerNorm2d(dim)
        self.drop_path = DropPath(0.1) if 0.1 > 0. else nn.Identity()
        
        self.layer_scale1 = nn.Identity()
        self.layer_scale2 = nn.Identity()
        self.layer_scale3 = nn.Identity()
        self.layer_scale4 = nn.Identity()
        self.layer_scale5 = nn.Identity()
        self.layer_scale6 = nn.Identity()
        
    def forward(self, x, y):
        # x B C H W
        
        shortcut = x
        shortcut2 = y

        input1 = self.norm1(x)
        input2 = self.norm1_(y)
        input1 = self.attn(input1, input1)
        input2 = self.attn2(input2, input2)

        input1 = self.drop_path(self.layer_scale1(input1)) + shortcut
        input2 = self.drop_path(self.layer_scale2(input2)) + shortcut2

        x_ = self.norm2(input1)
        y_ = self.norm2_(input2)

        input1 = self.cross_attn(x_, y_)
        input2 = self.cross_attn2(y_, x_)

        input1 = self.drop_path(self.layer_scale3(input1)) + x_
        input2 = self.drop_path(self.layer_scale4(input2)) + y_

        input1 = self.drop_path(self.layer_scale5(self.mlp(input1))) + input1
        input2 = self.drop_path(self.layer_scale6(self.mlp2(input2))) + input2

        input1 = self.norm3(input1)
        input2 = self.norm3_(input2)

        return input1, input2
    


@MODELS.register()
class SpectralAttentionv3(nn.Module):
    def __init__(self, dim, depth, num_heads,shared=True,attn= 'D',mlp_ratio=2, groups=4,
                  stride=4,scale=2.,nat_ks=3, kernel_size=5, dilation=1, attn_drop=0.1, proj_drop=0.1, size=80, **kwargs):
        super(SpectralAttentionv3, self).__init__()
        self.depth = depth
        blks = []
        self.shared = shared
        if not shared:
            for i in range(depth):
                blk = AttentionBlock(dim, num_heads, groups, stride,mlp_ratio, attn ,scale,
                                      nat_ks, kernel_size, dilation, attn_drop, proj_drop, size)
                blks.append(blk)
            self.blocks = nn.ModuleList(blks)
        else:
            blk = AttentionBlock(dim, num_heads, groups, stride,mlp_ratio, attn ,
                                 scale, nat_ks, kernel_size, dilation, attn_drop, proj_drop, size)
            self.blocks = blk
    
    def forward(self, x, y):
        # x B C H W
        if self.shared:
            for i in range(self.depth):
                x,y = self.blocks(x,y)
        else:
            for blk in self.blocks:
                x, y = blk(x,y)
        return x, y

@MODELS.register()
class SpectralAttentionv4(nn.Module):
    def __init__(self, dim, depth, num_heads, groups=4, stride=4, mlp_ratio=2,scale=2.,
                 nat_ks=3, kernel_size=5, dilation=1, attn_drop=0.1, proj_drop=0.1,shared=True, size=80,**kwargs):
        super(SpectralAttentionv4, self).__init__()
        self.depth = depth
        self.shared = shared
        if not shared:
            blks = []
            for i in range(depth):
                blk = AttentionBlock2(dim, num_heads, groups, stride, mlp_ratio,scale, nat_ks, kernel_size, dilation, attn_drop, proj_drop, size)
                blks.append(blk)
            self.blocks = nn.ModuleList(blks)
        else:
            blk = AttentionBlock2(dim, num_heads, groups, stride, mlp_ratio,scale, nat_ks, kernel_size, dilation, attn_drop, proj_drop, size)
            self.blocks = blk
    
    def forward(self, x, y):
        # x B C H W
        if self.shared:
            for i in range(self.depth):
                x,y = self.blocks(x,y)
        else:
            for blk in self.blocks:
                x,y = blk(x,y)
        return x, y




class SpectralFusion(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                    dim = 256,
                    version = 'v4',
                    fusion_config = dict(
                        dim=256,
                        depth=1,
                        num_heads=8,
                        shared=True,
                        attn= 'D',
                        mlp_ratio=2,
                        groups=4,
                        stride=4,
                        scale=2.,
                        nat_ks=3,
                        kernel_size=5,
                        dilation=1,
                        attn_drop=0.1,
                        proj_drop=0.1,
                        size=80,
                    )
                ):
        super().__init__()
        self.point_wise = nn.Conv2d(dim*2, dim,1) 
        # build blocks
        if version == 'v3':
            self.attention = SpectralAttentionv3(**fusion_config)
        elif version == 'v4':
            self.attention = SpectralAttentionv4(**fusion_config)
        
        self.version = version
        
        
    def forward(self,rgb,ir):
        """ Forward function.
        Args:
            query: Input feature, tensor size (B, C,H,W).
        """
        rgb_f, ir_f = self.attention(rgb,ir)        
        feature_map = torch.cat((rgb+rgb_f,ir+ir_f),dim=1)
        return self.point_wise(feature_map)  #, rgb_f, ir_f



@MODELS.register()
class SpectralAttentionFusionNeck(nn.Module):
    def __init__(self,
                 in_channels = [256, 256, 256, 256],
                 version = 'v4',
                 depths=[1, 1, 1, 1],
                 num_heads=8,
                 shared=True,
                 attns=['D', 'D', 'D', 'D'],
                 mlp_ratio=4,
                groups=[4, 4, 4, 4],
                stride=[8, 4, 2, 1],
                scale=5.,
                nat_ks=[5, 5, 5, 5],
                dat_ks=[9, 7, 5, 3],
                feat_strides=[8, 16, 32, 64],
                attn_drop=0.,
                proj_drop=0.,
                eval_size=640,
                qkv_bias=False,
                drop_path=0.2,
                **kwargs
                ):
        super().__init__()
        self.in_channels = in_channels
        self.version = version
        self.fusion_layers = nn.ModuleList()
        for idx, in_ch in enumerate(in_channels):
            fusion_layer = SpectralFusion(
                dim=in_ch,
                version=version,
                fusion_config=dict(
                    dim=in_ch,
                    depth=depths[idx],
                    num_heads=num_heads,
                    shared=shared,
                    attn=attns[idx],
                    mlp_ratio=mlp_ratio,
                    groups=groups[idx],
                    stride=stride[idx],
                    scale=scale,
                    nat_ks=nat_ks[idx],
                    kernel_size=dat_ks[idx],
                    dilation=1,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    size=eval_size // feat_strides[idx],
                    qkv_bias=qkv_bias,
                    drop_path=drop_path,
                    **kwargs
                )
            )

            self.fusion_layers.append(fusion_layer)
        
    def forward(self, rgb_feats, ir_feats):
        fused_feats = []
        for i in range(len(self.in_channels)):
            fused_feat = self.fusion_layers[i](rgb_feats[i], ir_feats[i])
            fused_feats.append(fused_feat)
        return fused_feats
