import torch
import torch.nn as nn
import torch.nn.functional as F
from util import MODELS
from typing import List, Optional, Tuple, Dict, Any
import copy, random, itertools
import numpy as np
import math
from util.box_ops import box_cxcywh_to_xyxy,box_xyxy_to_cxcywh
from torch import Tensor
from ..layers.bricks import MLP
from ..layers.utils import inverse_sigmoid


__all__ = ['DINO']

@MODELS.register()
class DINO(nn.Module):
    def __init__(self,
        backbone: Dict[str, Any],
        position_encoding: Dict[str, Any],
        task_enconding: Dict[str, Any],
        neck: Optional[Dict[str, Any]],
        transformer: Dict[str, Any],
        fusion_neck: Optional[Dict[str, Any]] = None,
        meta_score: Optional[Dict[str, Any]] = None,
        episode_size: int = 4,
        num_episodes: int = 4,
        embed_dim: int = 256,
        num_classes: int = 80,
        num_queries: int = 900,
        criterion: Optional[Dict[str, Any]] = None,
        cdn_generator: Optional[Dict[str, Any]] = None,
        aux_loss: bool = True,
        proto_loss_weight: float = 1.0,
        test_cfg=dict(
            max_per_img=300,
            nms_iou_threshold=-1,
            )
        ):
        super().__init__()
        # define backbone and position embedding module
        self.backbone = MODELS.build(backbone)
        self.backbone_ir = MODELS.build(backbone)
        self.positional_encoding = MODELS.build(position_encoding)

        # define neck module
        self.neck = MODELS.build(neck) if neck is not None else None
        self.neck_ir = MODELS.build(neck) if neck is not None else None
        self.fusion_neck = MODELS.build(fusion_neck)

        # define transformer module
        self.transformer = MODELS.build(transformer)
        # define meta score module
        self.meta_score = MODELS.build(meta_score) if meta_score is not None else None

        # define classification head and box head
        self.class_embed = nn.Linear(embed_dim, episode_size)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.gt_num_classes = num_classes
        self.num_classes = episode_size

        # number of dynamic anchor boxes and embedding dimension
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.criterion = MODELS.build(criterion) if criterion is not None else None
        if cdn_generator is not None:
            cdn_generator['embed_dims'] = embed_dim
            cdn_generator['num_classes'] = self.num_classes
            cdn_generator['num_matching_queries'] = num_queries
        self.cdn_generator = MODELS.build(cdn_generator) if cdn_generator is not None else None
        self.aux_loss = aux_loss
        self.test_cfg = test_cfg
        self.episode_size = episode_size
        self.num_episodes = num_episodes

        # initialize weights
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(episode_size) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)
        for _, neck_layer in self.neck_ir.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers + 1
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

        # two-stage
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # hack implementation for two-stage
        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)

        #self.query_embedding = QueryEncoding(d_model=self.embed_dims, max_len=self.num_queries, dropout=0.0)
        self.pseudo_class = MODELS.build(task_enconding)
        self.weight_prototypes = proto_loss_weight
        base_weight_dict = copy.deepcopy(self.criterion.weight_dict)
        if self.aux_loss:
            weight_dict = self.criterion.weight_dict
            aux_weight_dict = {}
            aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
            for i in range(self.transformer.decoder.num_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
            weight_dict.update(aux_weight_dict)
            self.criterion.weight_dict = weight_dict


    def extract_features(self, rgb, ir):
        rgb_feats = self.backbone(rgb)
        ir_feats = self.backbone_ir(ir)
        if self.neck is not None:
            rgb_feats = self.neck(rgb_feats)
            ir_feats = self.neck_ir(ir_feats)
        return rgb_feats, ir_feats
    

    def pre_transformer(self, rgb_feats, ir_feats, 
                        support=False, support_shape=None,batch_data_samples=None):
        bs = rgb_feats[0].size(0)
        if support:
            assert support_shape is not None, "Support shape must be provided when processing support features"
            same_shape_flag = True
            input_img_h, input_img_w = support_shape
            img_shape_list = [(input_img_h, input_img_w) for _ in range(bs)]
        else:
            batch_input_shape = batch_data_samples[0]['metainfo']['batch_input_shape']
            input_img_h, input_img_w = batch_input_shape
            img_shape_list = [sample['metainfo']['img_shape'] for sample in batch_data_samples]
            same_shape_flag = all([
                s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list
            ])
        if same_shape_flag:
            masks = rgb_feats[0].new_zeros(
                (bs, input_img_h, input_img_w))
        else:
            masks = rgb_feats[0].new_ones(
                (bs, input_img_h, input_img_w))
            for img_id in range(bs):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_pos_embeds = []
        for feat_rgb, feat_ir in zip(rgb_feats, ir_feats):
            mlvl_masks.append(
                    F.interpolate(masks[None], size=feat_rgb.shape[-2:]).to(
                        torch.bool).squeeze(0))
            mlvl_pos_embeds.append(
                    self.positional_encoding(mlvl_masks[-1]))
        fused_feats = self.fusion_neck(rgb_feats, ir_feats) #if self.fusion_neck is not None else rgb_feats
        return fused_feats, mlvl_masks, mlvl_pos_embeds
    

    def forward_transformer(self, rgb, ir,
                            batch_data_samples, 
                            prototypes, 
                            support_class_ids,
                            device,
                            num_episodes=1):
        
        
        meta_episode_class = []
        dn_metas = []
        fused_feats, mlvl_masks, mlvl_pos_embeds = self.pre_transformer(rgb, ir,False,None,batch_data_samples)

        hs_list = []
        reference_list = []
        hs_enc_list = []
        ref_enc_list = []
        init_box_proposal_list = []
        for i in range(num_episodes):
            episode_prototypes = []
            pseudo_class =  self.pseudo_class(torch.zeros(self.episode_size, self.embed_dim, device=device))
            try:
                episode_class = support_class_ids[i*self.episode_size:(i+1)*self.episode_size]
            except:
                episode_class = support_class_ids[-self.episode_size:]
            meta_episode_class.append(episode_class)
            for l in range(len(prototypes)):
                lvl_ptpe = []
                for j in range(self.transformer.num_feature_levels):
                    try:
                        ptpe = prototypes[l][j][i*self.episode_size:(i+1)*self.episode_size] 
                    except:
                        ptpe = prototypes[l][j][-self.episode_size:]
                    lvl_ptpe.append(ptpe)
                episode_prototypes.append(lvl_ptpe)
            # replace the labels in the batch_data_samples with indexes in the episode of the classes
            if self.cdn_generator is not None and self.training:
                input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(batch_data_samples, episode_class, device)
            else:
                input_query_label, input_query_bbox, attn_mask = None, None, None
                dn_meta = None
            
            dn_metas.append(dn_meta)
            hs, reference, hs_enc, ref_enc, init_box_proposal = \
                self.transformer(fused_feats, mlvl_masks, episode_prototypes ,input_query_bbox, 
                                 mlvl_pos_embeds,input_query_label,attn_mask,pseudo_class)
            
            hs_list.append(hs)
            reference_list.append(reference)
            hs_enc_list.append(hs_enc)
            ref_enc_list.append(ref_enc)
            init_box_proposal_list.append(init_box_proposal)

        return hs_list, reference_list, hs_enc_list, \
            ref_enc_list, init_box_proposal_list, meta_episode_class, dn_metas


    @torch.no_grad()
    def sample_support_categories(self, batch_data_samples, support_data_samples):
        """Sample support categories for few-shot fine-tuning.

        Args:
            batch_data_samples (list[:obj:`DetDataSample`]): The batch data samples.
            support_data_samples (list[:obj:`DetDataSample`]): The support data samples.
        
        Returns:
            meta_support_rgb (Tensor): Support RGB images stacked.
            meta_support_ir (Tensor): Support IR images stacked.
            meta_support_targets (list[:obj:`DetDataSample`]): Support targets.
        """
        # Flatten support data
        support_targets = list(itertools.chain.from_iterable([t['data_samples'] for t in support_data_samples]))
        support_rgb = list(itertools.chain.from_iterable([t['inputs_rgb'] for t in support_data_samples]))
        support_ir = list(itertools.chain.from_iterable([t['inputs_ir'] for t in support_data_samples]))
        support_class_ids = [t['gt_instances']['labels'].item() for t in support_targets]

        # Positive and negative classes for this batch
        positive_labels = torch.cat([t['gt_instances']['labels'] for t in batch_data_samples], dim=0).unique()
        positive_labels_list = positive_labels.tolist()
        negative_labels_list = list(set(support_class_ids) - set(positive_labels_list))

        # Indexes of samples for positive and negative classes
        positive_label_indexes = [i for i, cls in enumerate(support_class_ids) if cls in positive_labels_list]
        negative_label_indexes = [i for i, cls in enumerate(support_class_ids) if cls in negative_labels_list]

        meta_support_rgb, meta_support_ir, meta_support_targets = [], [], []
                
        
        # Track usage of classes across episodes
        all_classes = set(support_class_ids)
        class_usage = {cls: 0 for cls in all_classes}
        for episode in range(self.num_episodes):
            min_pos = max(1, self.episode_size - len(negative_labels_list))
            max_pos = min(len(positive_labels_list), self.episode_size)
            NUM_POS = random.randint(min_pos, max_pos) if max_pos >= min_pos else min_pos
            NUM_NEG = self.episode_size - NUM_POS


            # Positive classes: weighted sampling by inverse usage
            pos_weights = np.array([1.0 / (1 + class_usage[cls]) for cls in positive_labels_list])
            pos_weights /= pos_weights.sum()
            selected_pos_classes = np.random.choice(positive_labels_list, size=NUM_POS, replace=False, p=pos_weights)
            pos_support_indexes = []
            for cls in selected_pos_classes:
                cls_indexes = [i for i in positive_label_indexes if support_class_ids[i] == cls]
                pos_support_indexes.append(random.choice(cls_indexes))
                class_usage[cls] += 1

            if len(negative_labels_list) > 0 and NUM_NEG > 0:
                # Negative classes: weighted sampling by inverse usage
                neg_weights = np.array([1.0 / (1 + class_usage[cls]) for cls in negative_labels_list])
                neg_weights /= neg_weights.sum()
                selected_neg_classes = np.random.choice(negative_labels_list, size=NUM_NEG, replace=False, p=neg_weights)
                neg_support_indexes = []
                for cls in selected_neg_classes:
                    cls_indexes = [i for i in negative_label_indexes if support_class_ids[i] == cls]
                    neg_support_indexes.append(random.choice(cls_indexes))
                    class_usage[cls] += 1
            else:
                neg_support_indexes = []

            # Combine and shuffle
            support_indexes = pos_support_indexes + neg_support_indexes
            random.shuffle(support_indexes)

            # Gather support data
            meta_support_rgb += [support_rgb[i] for i in support_indexes]
            meta_support_ir += [support_ir[i] for i in support_indexes]
            meta_support_targets += [support_targets[i] for i in support_indexes]

        # Convert to tensors
        meta_support_rgb = torch.stack(meta_support_rgb, dim=0)
        meta_support_ir = torch.stack(meta_support_ir, axis=0)

        # Clean up
        del support_rgb, support_ir, support_targets, support_data_samples
        return meta_support_rgb, meta_support_ir, meta_support_targets



    def compute_prototypes(self, support_rgb: Tensor, support_ir: Tensor,
                          support_targets) -> Tensor:
        """Compute prototypes from support set."""
        support_feats = self.extract_features(support_rgb, support_ir)
        ori_w = support_rgb.size(-1)
        scales = [feat.size(-1)/ori_w for feat in support_feats[0]]
        tsp = self.pseudo_class(torch.zeros(self.episode_size, self.embed_dim, device=support_rgb.device))
        prototypes = list()
        num_supp = support_rgb.size(0)
        srcs, mlvl_masks, mlvl_pos = self.pre_transformer(support_feats[0], support_feats[1], True, support_rgb.shape[-2:], None)
        for i in range(num_supp // self.episode_size):
            mask = [m[i*self.episode_size:(i+1)*self.episode_size] for m in mlvl_masks]
            feats = [src[i*self.episode_size:(i+1)*self.episode_size] for src in srcs]
            pos = [p[i*self.episode_size:(i+1)*self.episode_size] for p in mlvl_pos]
            ptpe = self.transformer.forward_support(srcs=feats,
                                                    masks=mask,
                                                    scales=scales,
                                                    pos_embeds=pos,
                                                    support_boxes=support_targets[i*self.episode_size: (i+1)*self.episode_size],
                                                    pseudo_class=tsp,
                                                    )
            prototypes.append(ptpe)

        final_prototypes = []
        for i in range(len(prototypes[0])):
            lvl_ptpe = []
            for j in range(self.transformer.num_feature_levels):
                lvl_ptpe.append(torch.cat([ccl[i][j] for ccl in prototypes], dim=0))
            final_prototypes.append(lvl_ptpe)   
        return final_prototypes

    def prepare_for_cdn(self, batch_data_samples, episode_class: Tensor, device: torch.device):
        meta_batch_data_samples = []
        for sample in batch_data_samples:
                new_sample = copy.deepcopy(sample)
                gt_instances = sample['gt_instances']
                gt_labels = gt_instances['labels']
                gt_bboxes = gt_instances['bboxes']

                new_gt_instances = dict()
                # Create mask: only keep labels present in the current episode
                mask = torch.isin(gt_labels, episode_class)
                if mask.sum() == 0:
                    # No labels in this sample belong to the episode
                    new_gt_instances['labels'] = torch.empty((0,), dtype=torch.long, device=device)
                    new_gt_instances['bboxes'] = torch.empty((0, 4), dtype=gt_bboxes.dtype, device=device)
                else:
                    filtered_labels = gt_labels[mask]
                    filtered_bboxes = gt_bboxes[mask]

                    # Remap labels to episode-local IDs
                    episode_map = {c.item(): i for i, c in enumerate(episode_class)}
                    remapped_labels = torch.tensor([episode_map[lbl.item()] for lbl in filtered_labels],
                                                dtype=torch.long, device=device)

                    new_gt_instances['labels'] = remapped_labels
                    new_gt_instances['bboxes'] = filtered_bboxes.to(device)
                
                new_sample['gt_instances'] = new_gt_instances
                meta_batch_data_samples.append(new_sample)
        return self.cdn_generator(meta_batch_data_samples)
    
    def forward(self, batch_inputs, batch_data_samples=None,
                 support_data_samples=None, prototypes=None, support_class_ids=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

       
        device = next(self.parameters()).device
        

        proto_logits = None
        if not self.training :
            assert support_data_samples is None, 'Prototypes and support data samples cannot be provided at the same time'
            prototypes = prototypes.to(device)
            support_class_ids = support_class_ids.to(device)  
            num_classes = prototypes.size(2)
            num_ep= math.ceil(num_classes / self.episode_size)            
        else:
            num_ep = self.num_episodes
            supports_rgb, support_ir, support_targets = \
                    self.sample_support_categories(batch_data_samples, support_data_samples)
            # print(support_targets)
            supports_rgb = supports_rgb.to(batch_inputs[0].device)
            support_ir = support_ir.to(batch_inputs[1].device)
            prototypes = self.compute_prototypes(supports_rgb,support_ir,support_targets)
            # for prototype loss 
            proto_logits = self.meta_score(prototypes)
            proto_logits  = proto_logits.view(-1, proto_logits.size(-1))
            support_class_ids = torch.cat([targ['gt_instances']['labels'] for targ in support_targets], dim=0).to(device)
            num_feature_levels = self.transformer.num_feature_levels
            proto_loss = F.cross_entropy(
            proto_logits, support_class_ids.unsqueeze(0).expand(num_feature_levels,-1).reshape(-1))
            proto_loss = {'proto_loss':proto_loss * self.weight_prototypes}
            
            
        
        samples, samples_ir = batch_inputs
        rgb, ir = self.extract_features(samples, samples_ir)
    

        
        hs_list, reference_list, hs_enc_list, \
            ref_enc_list, init_box_proposal_list, meta_episode_class, dn_metas = \
                self.forward_transformer(rgb, ir, batch_data_samples, prototypes, support_class_ids, device, num_ep)
        


        meta_outputs_class = []
        meta_outputs_coord = []
        meta_enc_outputs_class = []
        meta_enc_outputs_coord = []
        meta_interm_outputs_class = []
        meta_interm_outputs_coord = []
        meta_interm_bbox_proposal = []
        enc_bbox_embed = self.bbox_embed[-1]
        enc_class_embed = self.class_embed[-1]
        # make predictions for each episode and concatenate them together
        for episode_id in range(num_ep):
            reference = reference_list[episode_id]
            hs = hs_list[episode_id]
            hs_enc = hs_enc_list[episode_id]
            ref_enc = ref_enc_list[episode_id]
            init_box_proposal = init_box_proposal_list[episode_id]
            outputs_coord_list = []
            for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
                layer_delta_unsig = layer_bbox_embed(layer_hs)
                layer_outputs_unsig = layer_delta_unsig  + inverse_sigmoid(layer_ref_sig)
                layer_outputs_unsig = layer_outputs_unsig.sigmoid()
                outputs_coord_list.append(layer_outputs_unsig)
            outputs_coord_list = torch.stack(outputs_coord_list)        
            outputs_class = torch.stack([layer_cls_embed(layer_hs) for
                                        layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
            
            meta_outputs_class.append(outputs_class)
            meta_outputs_coord.append(outputs_coord_list)
            # for encoder output
            if hs_enc is not None:
                # prepare intermediate outputs
                interm_coord = ref_enc[-1]
                interm_class = enc_class_embed(hs_enc[-1])
                meta_interm_outputs_class.append(interm_class)
                meta_interm_outputs_coord.append(interm_coord)
                meta_interm_bbox_proposal.append(init_box_proposal)
                # prepare enc outputs
                if hs_enc.shape[0] > 1:
                    enc_outputs_coord = []
                    enc_outputs_class = []
                    for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) \
                        in enumerate(zip(enc_bbox_embed, enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
                        layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                        layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                        layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()
                        layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                        enc_outputs_coord.append(layer_enc_outputs_coord)
                        enc_outputs_class.append(layer_enc_outputs_class)

                    meta_enc_outputs_class.append(torch.stack(enc_outputs_class))
                    meta_enc_outputs_coord.append(torch.stack(enc_outputs_coord))

        meta_batch_gt_instances = []
        meta_batch_img_metas = []
        for data_sample in batch_data_samples:  # image-first
            for i in range(num_ep):       # episode-second
                episode_class_ids = meta_episode_class[i].cpu().numpy().tolist()
                meta_batch_img_metas.append(data_sample['metainfo'])
                gt_instances = data_sample['gt_instances']
                gt_labels = gt_instances['labels']
                # filter categories that are not in the current episode
                mask = torch.tensor([cat_id in episode_class_ids for cat_id in gt_labels], device=device)
                filtered_gt_instances = dict()
                filtered_gt_instances['bboxes'] = gt_instances['bboxes'][mask]
                filtered_gt_instances['labels'] = gt_instances['labels'][mask]
                meta_batch_gt_instances.append(filtered_gt_instances)

        num_q = self.num_queries if dn_metas[0] is None \
            else max([dn_meta['num_denoising_queries'] for dn_meta in dn_metas]) + self.num_queries
        num_enc = self.transformer.encoder.num_layers
        num_dec = self.transformer.decoder.num_layers
        final_meta_outputs_classes, final_meta_outputs_coords, \
            final_meta_enc_outputs_classes, final_meta_enc_outputs_coords, \
                final_meta_interm_outputs_classes, final_meta_interm_outputs_coords, \
                    final_meta_interm_bbox_proposal = self.arrange_predictions(meta_outputs_class,
                                                                             meta_outputs_coord, 
                                                                             meta_enc_outputs_class, 
                                                                             meta_enc_outputs_coord, 
                                                                             meta_interm_outputs_class, 
                                                                             meta_interm_outputs_coord, 
                                                                             meta_interm_bbox_proposal,
                                                                             hs_enc,
                                                                             meta_episode_class,
                                                                            batch_inputs[0].size(0), 
                                                                            num_dec, num_enc ,num_q, num_ep, device)

        # denoising postprocessing
        if dn_metas[0] is not None:
            final_meta_outputs_classes, final_meta_outputs_coords = self.dn_post_process(
                final_meta_outputs_classes, final_meta_outputs_coords , dn_metas, meta_episode_class
            )

        num_l, bs, num_ep, num_q, _ = final_meta_outputs_classes.shape
        final_meta_outputs_classes = \
            final_meta_outputs_classes.view(num_l, bs*num_ep, num_q, self.gt_num_classes)
        final_meta_outputs_coords = \
            final_meta_outputs_coords.view(num_l, bs*num_ep, num_q, 4)
        bs = batch_inputs[0].size(0)
        # prepare for loss computation
        output = {"pred_logits": final_meta_outputs_classes[-1],
                   "pred_boxes": final_meta_outputs_coords[-1]}
        output['activated_class_ids'] = torch.stack(meta_episode_class).unsqueeze(0).expand(bs, -1, -1).reshape(bs * num_ep, -1)
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(final_meta_outputs_classes, final_meta_outputs_coords)
            for aux_output in output['aux_outputs']:
                aux_output['activated_class_ids'] = \
                    torch.stack(meta_episode_class).unsqueeze(0).expand(bs, -1, -1).reshape(bs * num_ep, -1)


        output["enc_outputs"] = {"pred_logits": final_meta_interm_outputs_classes, 
                                 "pred_boxes": final_meta_interm_outputs_coords}
        output["enc_outputs"]['activated_class_ids'] = \
            torch.stack(meta_episode_class).unsqueeze(0).expand(bs, -1, -1).reshape(bs * num_ep, -1)
        

        if self.training:
            targets = self.prepare_targets(meta_batch_gt_instances, meta_batch_img_metas)
            # compute loss
            loss_dict = self.criterion(output, targets, dn_metas)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            loss_dict.update(proto_loss)
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.predict(box_cls, box_pred, batch_data_samples, bs, num_ep)
            return results
    
    def prepare_targets(self, batch_data_samples, batch_img_metas):
        """
        prepare targets for loss computation
        normalize gt_bboxes, etc...
        """
        targets = []
        for data_sample, img_meta in zip(batch_data_samples, batch_img_metas):
            gt_instances = data_sample#['gt_instances']
            gt_labels = gt_instances['labels']
            gt_bboxes = gt_instances['bboxes']
            img_h, img_w = img_meta['img_shape']
            boxes = box_xyxy_to_cxcywh(gt_bboxes)
            # normalize gt_bboxes
            gt_bboxes_normalized = boxes / torch.tensor([img_w, img_h, img_w, img_h], device=gt_bboxes.device)
            targets.append({'labels': gt_labels, 'boxes': gt_bboxes_normalized})
        return targets


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]
    
    def arrange_predictions(self,meta_outputs_class,
                             meta_outputs_coord, 
                             meta_enc_outputs_class, 
                             meta_enc_outputs_coord, 
                             meta_interm_outputs_class, 
                             meta_interm_outputs_coord, 
                             meta_interm_bbox_proposal,
                             hs_enc,
                             meta_episode_class,
                            batchsize, num_dec, num_enc, num_queries, num_episodes, device):
        
        final_meta_outputs_classes = torch.ones(num_dec, batchsize, num_episodes, num_queries, self.gt_num_classes, device=device) * (-999999.99)
        final_meta_outputs_coords = torch.zeros(num_dec, batchsize, num_episodes, num_queries, 4, device=device)

        final_meta_enc_outputs_classes = torch.ones(num_enc, batchsize, num_episodes, self.num_queries, self.gt_num_classes, device=device) * (-999999.99)
        final_meta_enc_outputs_coords = torch.zeros(num_enc, batchsize, num_episodes, self.num_queries, 4, device=device)
        final_meta_interm_outputs_classes = torch.ones(batchsize, num_episodes, self.num_queries, self.gt_num_classes, device=device) * (-999999.99)
        final_meta_interm_outputs_coords = torch.zeros(batchsize, num_episodes, self.num_queries, 4, device=device)
        final_meta_interm_bbox_proposal = torch.zeros(batchsize, num_episodes, self.num_queries, 4, device=device)

        class_ids_already_filled_in = []
        for episode in range(num_episodes):
            num_dec_queries = meta_outputs_class[episode].size(2)
            for idx, class_id in enumerate(meta_episode_class[episode]):
                if self.training or (class_id.item() not in class_ids_already_filled_in):
                    class_ids_already_filled_in.append(class_id.item())
                    final_meta_outputs_classes[:, :, episode, -num_dec_queries:, class_id] = meta_outputs_class[episode][:, :, :, idx]
                    final_meta_outputs_coords[:, :, episode, -num_dec_queries:, :] = meta_outputs_coord[episode][:, :, :, :]
                    if hs_enc is not None:
                        if hs_enc.shape[0] > 1:
                            final_meta_enc_outputs_classes[:, :, episode, :, class_id] = meta_enc_outputs_class[episode][:, :, :, idx]
                            final_meta_enc_outputs_coords[:, :, episode, :, :] = meta_enc_outputs_coord[episode][:, :, :, :]
                        final_meta_interm_outputs_classes[:, episode, :, class_id] = meta_interm_outputs_class[episode][ :, :, idx]
                        final_meta_interm_outputs_coords[ :, episode, :, :] = meta_interm_outputs_coord[episode][:, :, :]
                        final_meta_interm_bbox_proposal[:, episode, :, :] = meta_interm_bbox_proposal[episode][ :, :, :]
        
        """# simulate a batch size of num_episodes*batchsize
        final_meta_outputs_classes = final_meta_outputs_classes.view(num_enc, batchsize*num_episodes, num_queries, self.gt_num_classes)
        final_meta_outputs_coords = final_meta_outputs_coords.view(num_enc, batchsize*num_episodes, num_queries, 4)"""
        if hs_enc is not None:
            if hs_enc.shape[0] > 1:
                final_meta_enc_outputs_classes = final_meta_enc_outputs_classes.view(num_enc, batchsize*num_episodes, self.num_queries, self.gt_num_classes)
                final_meta_enc_outputs_coords = final_meta_enc_outputs_coords.view(num_enc, batchsize*num_episodes, self.num_queries, 4)
            final_meta_interm_outputs_classes = final_meta_interm_outputs_classes.view(batchsize*num_episodes, self.num_queries, self.gt_num_classes)
            final_meta_interm_outputs_coords = final_meta_interm_outputs_coords.view(batchsize*num_episodes, self.num_queries, 4)
            final_meta_interm_bbox_proposal = final_meta_interm_bbox_proposal.view(batchsize*num_episodes, self.num_queries, 4)

        return final_meta_outputs_classes, final_meta_outputs_coords, \
                final_meta_enc_outputs_classes, final_meta_enc_outputs_coords, \
                    final_meta_interm_outputs_classes, final_meta_interm_outputs_coords, \
                    final_meta_interm_bbox_proposal


    def dn_post_process(self, outputs_class, outputs_coord, dn_metas, meta_episode_class):
        if True:
            #padding_size = dn_metas["single_padding"] * dn_metas["dn_num"]
            max_num_dn = max([dn_meta['num_denoising_queries'] for dn_meta in dn_metas])
            final_outputs_class = []
            final_outputs_coord = []
            for idx, dn_meta in enumerate(dn_metas):
                num_dn = dn_meta['num_denoising_queries']
                padded_dn = max_num_dn - num_dn
                class_o = outputs_class[:, :, idx, -self.num_queries:, :]
                coord_o = outputs_coord[:, :, idx, -self.num_queries:, :]
                known_class_o = outputs_class[:, :, idx, padded_dn:max_num_dn, :]
                known_coord_o = outputs_coord[:, :, idx, padded_dn:max_num_dn, :]
                final_outputs_class.append(class_o)
                final_outputs_coord.append(coord_o)
                out = {"pred_logits": known_class_o[-1], "pred_boxes": known_coord_o[-1]}
                out['activated_class_ids'] = meta_episode_class[idx].unsqueeze(0).expand(known_class_o.size(1), -1)
                if self.aux_loss:
                    out["aux_outputs"] = self._set_aux_loss(known_class_o, known_coord_o)
                    for aux_output in out['aux_outputs']:
                        aux_output['activated_class_ids'] = meta_episode_class[idx].unsqueeze(0).expand(known_class_o.size(1), -1)
                dn_metas[idx]["output_known_lbs_bboxes"] = out
            outputs_class = torch.stack(final_outputs_class, dim=2)
            outputs_coord = torch.stack(final_outputs_coord, dim=2)
        return outputs_class, outputs_coord
    
    def predict(self, box_cls, box_pred, batch_data_samples, batchsize, num_episode=1):
        num_queries = box_cls.shape[1]
        num_classes = box_cls.shape[-1]
        box_cls = box_cls.view(batchsize, num_episode * num_queries, num_classes)
        box_pred = box_pred.view(batchsize, num_episode * num_queries, 4)
        assert len(box_cls) == len(batch_data_samples), "Batch size of predictions and data samples must match"
        metainfos = [data_sample['metainfo'] for data_sample in batch_data_samples]
        results = []
        max_pred = self.test_cfg['max_per_img']
        # box_cls.shape: 1, 300, 80
        # box_pred.shape: 1, 300, 4
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1),max_pred , dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]
        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))


        for i, (scores_per_image, labels_per_image, box_pred_per_image, metainfo) in enumerate(
            zip(scores, labels, boxes, metainfos)
        ):
            result = dict()
            img_shape = metainfo['img_shape']
            det_bboxes = box_cxcywh_to_xyxy(box_pred_per_image)
            det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
            det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
            det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
            det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
            scale_factor = metainfo['scale_factor']
            det_bboxes /= det_bboxes.new_tensor(
                scale_factor).repeat((1, 2))
            result['scores'] = scores_per_image
            result['labels'] = labels_per_image
            result['bboxes'] = det_bboxes
            results.append(result)
        
        for result, data_sample in zip(results, batch_data_samples):
            data_sample['pred_instances'] = result
        return batch_data_samples