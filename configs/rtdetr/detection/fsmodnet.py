mean = [0., 0., 0.]
std = [255., 255., 255.]
num_classes = 80
num_episodes = 5
model = dict(
    type='RTDETR',
    class_embed=dict(
        type='ContrastiveEmbed',
        max_text_len=num_classes,
    ),
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
    language_model=dict(
        type='HuggingCLIPLanguageBackbone',
        model_name='openai/clip-vit-base-patch32',
        add_mask=True,
        max_length=256,
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=3,
        bias=False,
        act_cfg=None,
        norm_cfg=dict(type='BN', num_features=256),
    ),
    cdn_generator=dict(
        type='CdnQueryGenerator',
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    transformer=dict(
        type='RTDETRTransformer',
        encoder=dict(
                    type='MST_Encoder',
             ),
        decoder=dict(
                    type='RTDETRTransformerDecoder',
                    d_model=256, 
                    attn_dropout=0., 
                    ffn_dropout=0.,
                    n_levels=6, n_heads=8, n_points=4,
                    d_ffn=1024, num_layers=6, activation='relu',
                    return_intermediate=True, checkpoint=False
        ),
        learned_init_query=True, 
        num_queries=300, 
        ),
    num_episodes=num_episodes,
    embed_dim=256,
    num_classes=num_classes,
    num_queries=300,
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




