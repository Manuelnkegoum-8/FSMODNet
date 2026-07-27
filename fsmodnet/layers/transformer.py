# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import math, random
import copy
from typing import Optional

import torch
from torch import nn, Tensor
from .utils import inverse_sigmoid
from .utils import gen_encoder_output_proposals,_get_activation_fn, gen_sineembed_for_position
from ..ops.modules import MSDeformAttn
from torchvision.ops import roi_align
from .encoding import QuerySupportAttention, QueryEncoding
from util import MODELS
from fairscale.nn.checkpoint import checkpoint_wrapper
from .bricks import MLP


__all__ = ['TransformerEncoder', 'TransformerDecoder',  'DeformableTransformer']

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,attn_dropout=0.1,
                 n_levels=4, n_heads=8, n_points=4,
                 ffn=dict(type='FFN', d_model=256, d_ffn=1024, dropout=0.1, activation='relu'),
                 Qsa=dict(type='QuerySupportAttention', num_heads=1, d_model=256)
                 ):
        super().__init__()
        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(attn_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.d_model = d_model
        # ffn
        self.ffn = MODELS.build(ffn)
        self.norm2 = nn.LayerNorm(d_model)
        self.qsa = Qsa
        # Query support attention
        if Qsa is not None:
            multi_level_cross_attention = []
            for _ in range(n_levels):
                qa_layer = MODELS.build(Qsa)
                multi_level_cross_attention.append(qa_layer)
            self.query_support = nn.ModuleList(multi_level_cross_attention)
            
            

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos


    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None,prototypes=None,pseudo_class=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        if self.qsa:
            #print('pseudo class in encoder layer:', pseudo_class)
            ### CROSS ATTENTION WITH PROTOTYPE
            if prototypes is not None:
                updated_memory = src.clone()  # Create a copy of the memory tensor
                for lvl, spatial_shape in enumerate(spatial_shapes):
                    #h, w = spatial_shape
                    if lvl + 1 < len(level_start_index):
                        mem = src[:, level_start_index[lvl]:level_start_index[lvl+1], :]
                    else:
                        mem = src[:, level_start_index[lvl]:, :]
                    ptpe = prototypes[lvl] # episode_size x d_model
                    ptpe = ptpe.unsqueeze(0).repeat(mem.shape[0], 1, 1) # bs x episode_size x d_model
                    mem, _ = self.query_support[lvl](mem,ptpe,ptpe, pseudo_class)
                    if lvl + 1 < len(level_start_index):
                        updated_memory[:, level_start_index[lvl]:level_start_index[lvl+1], :] = mem
                    else:
                        updated_memory[:, level_start_index[lvl]:, :] = mem

                src = updated_memory  # Update the original memory tensor with the new values
                
        # ffn
        src = self.ffn(src)
        src = self.norm2(src)
        return src
    
    def roi_align(self, feature_maps, support_boxes, spatial_scale=1/32, spatial_shapes=None):
        # feature_maps: [bs, H*W , C]
        # support_boxes: [bs, num_rois, 4]
        h,w = spatial_shapes
        bs = feature_maps.size(0)
        feature_maps = feature_maps.permute(0, 2, 1).reshape(bs, -1,h,w)
        boxes = [bbox.cuda() for bbox in support_boxes]
        # roi align
        roi_features = roi_align(feature_maps, boxes, (7,7), spatial_scale, aligned = True)
        prototypes = roi_features.mean(dim=(-2,-1))  # Average pool the ROI features
        return prototypes
        
    def forward_support(self, src, pos, reference_pts, spatial_shapes, 
                        level_start_index, key_padding_mask=None, scales = None, support_boxes=None,pseudo_class=None):
        
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_pts, src, spatial_shapes, level_start_index, key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        bs = len(support_boxes)
        support_boxes = [t['gt_instances']['bboxes'] for t in support_boxes]
        # perform ROI align per level dim=0
        multi_level_prototypes = []
        updated_mem = src.clone()
        for lvl, spatial_shape in enumerate(spatial_shapes):
            if lvl + 1 < len(level_start_index):
                mem = src[:, level_start_index[lvl]:level_start_index[lvl+1], :]
            else:
                mem = src[:, level_start_index[lvl]:, :]
            prototypes = self.roi_align(mem, support_boxes, spatial_scale=scales[lvl], spatial_shapes=spatial_shape)
            if self.qsa:
                ptpe = prototypes.unsqueeze(0).repeat(mem.shape[0], 1, 1) # bs x episode_size x d_model
                mem, _ = self.query_support[lvl](mem,ptpe,ptpe, pseudo_class)

                if lvl + 1 < len(level_start_index):
                    updated_mem[:, level_start_index[lvl]:level_start_index[lvl+1], :] = mem
                else:
                    updated_mem[:, level_start_index[lvl]:, :] = mem
            # reshape to [bs, num_class, d_model]
            #prototypes = prototypes.reshape(bs, -1, self.d_model)
            multi_level_prototypes.append(prototypes)
        
        src = self.ffn(updated_mem)
        src = self.norm2(src)
        return src, multi_level_prototypes
    



class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, attn_dropout=0.1,
                 n_levels=4, n_heads=8, n_points=4,
                 ffn=dict(type='FFN', d_model=256, d_ffn=1024, dropout=0.1, activation='relu'),
                 ):
        super().__init__()
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(attn_dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout)
        self.dropout2 = nn.Dropout(attn_dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = MODELS.build(ffn)
        self.norm3 = nn.LayerNorm(d_model)


    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None, # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None, # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None, # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None, # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None, # num_levels
                memory_spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None, # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None, # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None, # mask used for cross-attention
            ):

        # self attention
        q = k = self.with_pos_embed(tgt, tgt_query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                               tgt_reference_points.transpose(0, 1).contiguous(),
                               memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt = self.ffn(tgt)
        tgt = self.norm3(tgt)
        return tgt



@MODELS.register()
class TransformerEncoder(nn.Module):
    def __init__(self, 
            d_model=256,
            attn_dropout=0.1,
            ffn_dropout=0.1,
            d_ffn=1024,
            n_levels=4, n_heads=8, n_points=4,
            num_layers=6,
            activation='relu',
            post_norm: bool = False,
            checkpoint=False,
            qsa_attn=dict(type='QuerySupportAttention', num_heads=1, dim=256)

        ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i!=0:
                qsa = None
            else:
                qsa = qsa_attn
            encoder_layer = DeformableTransformerEncoderLayer(
                    d_model=d_model, attn_dropout=attn_dropout,
                    n_levels=n_levels, n_heads=n_heads, n_points=n_points,
                    ffn=dict(type='FFN', d_model=d_model, d_ffn=d_ffn, dropout=ffn_dropout, activation=activation),
                    Qsa=qsa
                        )
            layer = copy.deepcopy(encoder_layer)
            self.layers.append(layer)
        self.num_layers = num_layers
        self.post_norm = post_norm
        if post_norm:
            self.norm = nn.LayerNorm(d_model)
        self.checkpoint = checkpoint
        if checkpoint:
            for layer in self.layers:
                layer = checkpoint_wrapper(layer)
                

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    

    def forward_support(self, src, pos, spatial_shapes, level_start_index, valid_ratios, key_padding_mask,
                        scales=None, support_boxes=None,pseudo_class_embed=None):
        output = src
        prototypes = []
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for layer_id, layer in enumerate(self.layers):
           output, proto = layer.forward_support(src=output, pos=pos, reference_pts=reference_points, spatial_shapes=spatial_shapes,
                                   level_start_index=level_start_index, key_padding_mask=key_padding_mask, scales=scales,
                                   support_boxes = support_boxes, pseudo_class=pseudo_class_embed)
           prototypes.append(proto)
        
        return prototypes

    def forward(self, 
            src: Tensor, 
            pos: Tensor, 
            spatial_shapes: Tensor, 
            level_start_index: Tensor, 
            valid_ratios: Tensor, 
            key_padding_mask: Tensor,
            ref_token_index: Optional[Tensor]=None,
            ref_token_coord: Optional[Tensor]=None,
            prototypes: Optional[Tensor]=None,
            pseudo_class_embed: Optional[Tensor]=None,
            ):

        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        reference_points = reference_points.to(src.dtype)
        # main process
        for layer_id, layer in enumerate(self.layers):
            
            ptpe = prototypes[layer_id] if prototypes is not None else None
            output = layer(src=output, pos=pos, reference_points=reference_points, spatial_shapes=spatial_shapes,
                                   level_start_index=level_start_index, 
                                   key_padding_mask=key_padding_mask, prototypes=ptpe, pseudo_class=pseudo_class_embed)  
                  

        if self.post_norm:
            output = self.norm(output)
        intermediate_output = intermediate_ref = None
    
        return output, intermediate_output,  intermediate_ref

@MODELS.register()
class TransformerDecoder(nn.Module):
    def __init__(self, 
            d_model=256, 
            attn_dropout=0.1, 
            ffn_dropout=0.1,
            n_levels=4, n_heads=8, n_points=4,
            d_ffn=1024, num_layers=6, activation='relu',
            return_intermediate=False, checkpoint=False):
        super().__init__()
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        
        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        
        self.ref_anchor_head = None
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=d_model, attn_dropout=attn_dropout,
            n_levels=n_levels, n_heads=n_heads, n_points=n_points,
            ffn=dict(type='FFN', d_model=d_model, d_ffn=d_ffn, dropout=ffn_dropout, activation=activation)
        )
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.checkpoint = checkpoint
        if checkpoint:
            for layer in self.layers:
                layer = checkpoint_wrapper(layer)
        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None, # num_queries, bs, 2
                # for memory
                level_start_index: Optional[Tensor] = None, # num_levels
                spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None,
                
                ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]  

        for layer_id, layer in enumerate(self.layers):
            
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                            * torch.cat([valid_ratios, valid_ratios], -1)[None, :] # nq, bs, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :]) # nq, bs, 256*2 

            # conditional query
            query_pos = self.ref_point_head(query_sine_embed) # nq, bs, 256
            
            output = layer(
                    tgt = output,
                    tgt_query_pos = query_pos,
                    tgt_query_sine_embed = query_sine_embed,
                    tgt_key_padding_mask = tgt_key_padding_mask,
                    tgt_reference_points = reference_points_input,

                    memory = memory,
                    memory_key_padding_mask = memory_key_padding_mask,
                    memory_level_start_index = level_start_index,
                    memory_spatial_shapes = spatial_shapes,
                    memory_pos = pos,

                    self_attn_mask = tgt_mask,
                    cross_attn_mask = memory_mask
                )

            # iter update
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_id](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            intermediate.append(self.norm(output))
            ref_points.append(new_reference_points)            

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]
        ]


@MODELS.register()
class DeformableTransformer(nn.Module):
    def __init__(self, encoder, 
                 decoder, 
                 learned_init_query=False, 
                 num_queries=300, 
                 num_feature_levels=4):
        super().__init__()
        self.encoder = MODELS.build(encoder)
        self.decoder = MODELS.build(decoder)
        self.learned_init_query = learned_init_query
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        d_model = self.encoder.layers[0].d_model
        self.d_model = d_model
        if num_feature_levels > 1:
            self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if learned_init_query:
            self.tgt_embed = QueryEncoding(d_model=d_model, max_len=self.num_queries)
        else:
            self.tgt_embed = None
        
        self.enc_output = nn.Linear(d_model, d_model)
        self.enc_output_norm = nn.LayerNorm(d_model)
        
        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)


    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)
        
        if self.random_refpoints_xy:

            self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

    def forward(self, srcs, masks, prototypes, refpoint_embed, pos_embeds, tgt, attn_mask=None, pseudo_class=None):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer
            
        """

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            mask = mask.flatten(1)                              # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c 
        mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # bs, \sum{hxw}, c 
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None

        #########################################################
        # Begin Encoder
        #########################################################

        memory, enc_intermediate_output, enc_intermediate_refpoints = self.encoder(
                src_flatten, 
                pos=lvl_pos_embed_flatten, 
                level_start_index=level_start_index, 
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                key_padding_mask=mask_flatten,
                ref_token_index=enc_topk_proposals, # bs, nq 
                ref_token_coord=enc_refpoint_embed, # bs, nq, 4
                prototypes=prototypes,
                pseudo_class_embed=pseudo_class
                )
      
        input_hw = None
        output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes, input_hw)
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        enc_outputs_class_unselected = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unselected = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals # (bs, \sum{hw}, 4) unsigmoid
        topk = self.num_queries
        topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1] # bs, nq

        # gather boxes
        refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid
        refpoint_embed_ = refpoint_embed_undetach.detach()
        init_box_proposal = torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid() # sigmoid

        # gather tgt
        tgt_undetach = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model))
        if self.learned_init_query:
                tgt_embed = self.tgt_embed()
                tgt_ = tgt_embed[:, None, :].repeat(1, bs, 1).transpose(0, 1) # nq, bs, d_model
        else:
                tgt_ = tgt_undetach.detach()

        if refpoint_embed is not None:
            refpoint_embed=torch.cat([refpoint_embed,refpoint_embed_],dim=1)
            tgt=torch.cat([tgt,tgt_],dim=1)
        else:
            refpoint_embed,tgt=refpoint_embed_,tgt_


        hs, references = self.decoder(
                tgt=tgt.transpose(0, 1), 
                memory=memory.transpose(0, 1), 
                memory_key_padding_mask=mask_flatten, 
                pos=lvl_pos_embed_flatten.transpose(0, 1),
                refpoints_unsigmoid=refpoint_embed.transpose(0, 1), 
                level_start_index=level_start_index, 
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,tgt_mask=attn_mask) 
        
        hs_enc = tgt_undetach.unsqueeze(0)
        ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        
        return hs, references, hs_enc, ref_enc, init_box_proposal

    def forward_support(self,srcs, masks, pos_embeds, scales=None, support_boxes=None, pseudo_class=None):
        """
        Input:
            - srcs: List of multi features [bs, ci, hi, wi]
            - masks: List of multi masks [bs, hi, wi]
            - refpoint_embed: [bs, num_dn, 4]. None in infer
            - pos_embeds: List of multi pos embeds [bs, ci, hi, wi]
            - tgt: [bs, num_dn, d_model]. None in infer
            
        """
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape

            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            mask = mask.flatten(1)                              # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)    # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c 
        mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # bs, \sum{hxw}, c 
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)

        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        #########################################################
        # Begin Encoder
        #########################################################
        prototypes = self.encoder.forward_support(
                src_flatten, 
                pos=lvl_pos_embed_flatten, 
                level_start_index=level_start_index, 
                spatial_shapes=spatial_shapes,
                valid_ratios=valid_ratios,
                key_padding_mask=mask_flatten,
                scales=scales,
                support_boxes=support_boxes,
                pseudo_class_embed=pseudo_class
                )
        # prototypes [num_encoder_layer,num_level,num_classes,d_model]
        return prototypes





