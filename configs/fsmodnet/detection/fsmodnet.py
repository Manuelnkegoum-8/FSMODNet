mean = [0., 0., 0.]
std = [255., 255., 255.]
num_classes = 80
episode_size = 5
num_episodes = 5
model = dict(
    type='DINO',
    backbone=dict(
        type='PResNet',
        depth=50,
        num_stages=4,
        return_idx=(1, 2, 3),
        freeze_at=0,
        freeze_norm=True,
        checkpoint_dir='../coco_pretrained_weights/',
        pretrained=True
        ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=4,
        bias=True,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32, num_channels=256),
    ),
    fusion_neck=dict(
                type='SpectralAttentionFusionNeck',
                in_channels = [256, 256, 256, 256],
                 version = 'v4',
                 depths=[1, 1, 1, 1],
                 num_heads=8,
                 shared=True,
                 attns=['L', 'L', 'L', 'L'],
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
    ),
     cdn_generator=dict(
        type='TaskCdnQueryGenerator',
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=300)),  # TODO: half num_dn_queries
    position_encoding=dict(
        type='PositionEmbeddingSine',
        num_pos_feats=128,
        normalize=True,
        offset=0,
        temperature=20,
    ),
    task_enconding=dict(
        type='TaskPositionalEncoding',
        d_model=256,
        max_len=episode_size,
        dropout=0.0),
    transformer=dict(
        type='DeformableTransformer',
        encoder=dict(
                    type='TransformerEncoder',
                    d_model=256,
                    attn_dropout=0.,
                    ffn_dropout=0.,
                    d_ffn=2048,
                    n_levels=4, n_heads=8, n_points=4,
                    num_layers=6,
                    activation='relu',
                    post_norm=False,
                    checkpoint=False,
                qsa_attn=dict(type='QuerySupportAttention', num_heads=1, dim=256)
             ),
        decoder=dict(
                    type='TransformerDecoder',
                    d_model=256, 
                    attn_dropout=0., 
                    ffn_dropout=0.,
                    n_levels=4, n_heads=8, n_points=4,
                    d_ffn=2048, num_layers=6, activation='relu',
                    return_intermediate=True, checkpoint=False
        ),
        learned_init_query=True, 
        num_queries=900, 
        num_feature_levels=4
        ),
    episode_size=episode_size,
    num_episodes=num_episodes,
    embed_dim=256,
    num_classes=num_classes,
    num_queries=900,
    meta_score=dict(
            type='Multi_lvl_CosineLinear',
            in_features=256,
            out_features=num_classes,
            num_scales=4,
            scale=10.,
            bias=False
        ),
    criterion=dict(
        type='DINOCriterion',
        num_classes=num_classes,
        matcher=dict(
            type='HungarianMatcher',
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 1,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_class_dn": 1,
            "loss_bbox_dn": 5.0,
            "loss_giou_dn": 2.0,
        },
        loss_class_type="varifocal_loss",
        alpha=0.75,
        gamma=2.0,
    ),
)




