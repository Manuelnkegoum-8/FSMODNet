mean = [122.74, 116.75, 104.09]
std = [68.49, 66.63, 70.32]
num_classes = 1
model = dict(
    type='RTDETR_UPN',
    class_embed=dict(
        type='ContrastiveEmbed',
        max_text_len=num_classes,
        use_contrastive=False,
    ),
     backbone=dict(
            type='ModifiedResNet',
            output_dim=1024,
            input_resolution=224,
            heads=32,
            layers= [3, 4, 6, 3],
            width=64,
            depth=50,
            pool_vec=False,
            create_att_pool=True,
            out_features=['res3', 'res4', 'res5'],
            pretrained='../coco_pretrained_weights/regionclip_pretrained-cc_rn50.pth'
        )   ,
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
        bias=True,
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
                    type='HybridEncoderv2',
             ),
        num_queries=300, 
        learnt_init_query=False, 
        ),
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




