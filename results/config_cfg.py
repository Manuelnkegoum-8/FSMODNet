mean = [123.675, 116.28, 103.53]
std = [58.395, 57.12, 57.375]
num_classes = 3
episode_size = 1
num_episodes = 3
model = dict(
    type='DINO',
    backbone=dict(
        type='TimmDINOModel',
        model='hf_hub:timm/vit_small_plus_patch16_dinov3.lvd1689m',
        n=1,
        reshape=True,
        return_class_token=False,
        feature_levels=[1, 2, 3]),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 384],
        out_channels=256,
        num_outs=4,
        bias=True,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32, num_channels=256)),
    fusion_neck=dict(
        type='SpectralAttentionFusionNeck',
        in_channels=[256, 256, 256, 256],
        version='v4',
        depths=[1, 1, 1, 1],
        num_heads=8,
        shared=True,
        attns=['L', 'L', 'L', 'L'],
        mlp_ratio=4,
        groups=[4, 4, 4, 4],
        stride=[8, 4, 2, 1],
        scale=5.0,
        nat_ks=[5, 5, 5, 5],
        dat_ks=[9, 7, 5, 3],
        feat_strides=[8, 16, 32, 64],
        attn_drop=0.0,
        proj_drop=0.0,
        eval_size=640,
        qkv_bias=False,
        drop_path=0.2),
    cdn_generator=dict(
        type='TaskCdnQueryGenerator',
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=300)),
    position_encoding=dict(
        type='PositionEmbeddingSine',
        num_pos_feats=128,
        normalize=True,
        offset=0,
        temperature=20),
    task_enconding=dict(
        type='TaskPositionalEncoding', d_model=256, max_len=1, dropout=0.0),
    transformer=dict(
        type='DeformableTransformer',
        encoder=dict(
            type='TransformerEncoder',
            d_model=256,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            d_ffn=2048,
            n_levels=4,
            n_heads=8,
            n_points=4,
            num_layers=6,
            activation='relu',
            post_norm=False,
            checkpoint=False,
            qsa_attn=dict(type='QuerySupportAttention', num_heads=1, dim=256)),
        decoder=dict(
            type='TransformerDecoder',
            d_model=256,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            n_levels=4,
            n_heads=8,
            n_points=4,
            d_ffn=2048,
            num_layers=6,
            activation='relu',
            return_intermediate=True,
            checkpoint=False),
        learned_init_query=True,
        num_queries=900,
        num_feature_levels=4),
    episode_size=1,
    num_episodes=3,
    embed_dim=256,
    num_classes=3,
    num_queries=900,
    meta_score=dict(
        type='Multi_lvl_CosineLinear',
        in_features=256,
        out_features=3,
        num_scales=4,
        scale=10.0,
        bias=False),
    criterion=dict(
        type='DINOCriterion',
        num_classes=3,
        matcher=dict(
            type='HungarianMatcher',
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type='focal_loss_cost',
            alpha=0.25,
            gamma=2.0),
        weight_dict=dict(
            loss_class=1,
            loss_bbox=5.0,
            loss_giou=2.0,
            loss_class_dn=1,
            loss_bbox_dn=5.0,
            loss_giou_dn=2.0),
        loss_class_type='varifocal_loss',
        alpha=0.75,
        gamma=2.0))
eval_size = (640, 640)
metainfo = dict(
    ALL_CLASSES=('bicycle', 'car', 'person'),
    BASE_CLASSES=('bicycle', 'car'),
    NOVEL_CLASSES=('person', ),
    mapping=dict(bicycle=1, car=2, person=3),
    palette=[(220, 20, 60), (0, 0, 142), (119, 11, 32)])
root = '../datasets/FLIR/'
train_pipeline = [
    dict(type='LoadMSImagesFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MSPhotoMetricDistortion', prob=0.5),
    dict(
        type='MSExpand',
        mean=[123.675, 116.28, 103.53],
        prob=0.5,
        ratio_range=(1, 4)),
    dict(type='MSRandomIoUCrop', prob=0.8),
    dict(type='FilterAnnotations', min_wh=(0.01, 0.01)),
    dict(type='MSResize', scale=(640, 640), keep_ratio=False),
    dict(type='MSHorizontalFlip', prob=0.5),
    dict(
        type='MSPackDetInputs',
        meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape',
                   'img_shape', 'scale_factor', 'flip', 'flip_direction'))
]
support_pipeline = [
    dict(type='LoadMSImagesFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MSHorizontalFlip', prob=0.5),
    dict(type='MSResize', scale=(640, 640), keep_ratio=False),
    dict(
        type='MSPackDetInputs',
        meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape',
                   'img_shape', 'scale_factor', 'flip', 'flip_direction'))
]
test_pipeline = [
    dict(type='LoadMSImagesFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MSFixScaleResize', scale=(640, 640), keep_ratio=False),
    dict(
        type='MSPackDetInputs',
        meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape',
                   'img_shape', 'scale_factor', 'flip', 'flip_direction',
                   'text', 'custom_entities'))
]
train_data = dict(
    type='Few_shot_CocoDetection_RGBT',
    root='../datasets/FLIR/',
    ann_file='few_shot/seed2/5.json',
    data_prefix=dict(img_rgb='JPEGImages/', img_ir='JPEGImages/'),
    transforms=[
        dict(type='LoadMSImagesFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='MSPhotoMetricDistortion', prob=0.5),
        dict(
            type='MSExpand',
            mean=[123.675, 116.28, 103.53],
            prob=0.5,
            ratio_range=(1, 4)),
        dict(type='MSRandomIoUCrop', prob=0.8),
        dict(type='FilterAnnotations', min_wh=(0.01, 0.01)),
        dict(type='MSResize', scale=(640, 640), keep_ratio=False),
        dict(type='MSHorizontalFlip', prob=0.5),
        dict(
            type='MSPackDetInputs',
            meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape',
                       'img_shape', 'scale_factor', 'flip', 'flip_direction'))
    ],
    support_transforms=[
        dict(type='LoadMSImagesFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='MSHorizontalFlip', prob=0.5),
        dict(type='MSResize', scale=(640, 640), keep_ratio=False),
        dict(
            type='MSPackDetInputs',
            meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape',
                       'img_shape', 'scale_factor', 'flip', 'flip_direction'))
    ],
    stage='few_shot_finetune',
    n_way=5,
    num_shots=1,
    min_area_support=256,
    key_rgb='file_name_RGB',
    key_ir='file_name_IR',
    meta_info=dict(
        ALL_CLASSES=('bicycle', 'car', 'person'),
        BASE_CLASSES=('bicycle', 'car'),
        NOVEL_CLASSES=('person', ),
        mapping=dict(bicycle=1, car=2, person=3),
        palette=[(220, 20, 60), (0, 0, 142), (119, 11, 32)]),
    test_mode=False,
    max_refetch=100,
    with_support=True)
val_data = dict(
    type='Few_shot_CocoDetection_RGBT',
    root='../datasets/FLIR/',
    ann_file='Coco_annotations/Val_Annotations.json',
    data_prefix=dict(img_rgb='JPEGImages/', img_ir='JPEGImages/'),
    transforms=[
        dict(type='LoadMSImagesFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='MSFixScaleResize', scale=(640, 640), keep_ratio=False),
        dict(
            type='MSPackDetInputs',
            meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape',
                       'img_shape', 'scale_factor', 'flip', 'flip_direction',
                       'text', 'custom_entities'))
    ],
    support_transforms=[
        dict(type='LoadMSImagesFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='MSHorizontalFlip', prob=0.5),
        dict(type='MSResize', scale=(640, 640), keep_ratio=False),
        dict(
            type='MSPackDetInputs',
            meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape',
                       'img_shape', 'scale_factor', 'flip', 'flip_direction'))
    ],
    stage='few_shot_finetune',
    n_way=5,
    num_shots=1,
    min_area_support=256,
    key_rgb='file_name_RGB',
    key_ir='file_name_IR',
    meta_info=dict(
        ALL_CLASSES=('bicycle', 'car', 'person'),
        BASE_CLASSES=('bicycle', 'car'),
        NOVEL_CLASSES=('person', ),
        mapping=dict(bicycle=1, car=2, person=3),
        palette=[(220, 20, 60), (0, 0, 142), (119, 11, 32)]),
    test_mode=True,
    with_support=False)
test_data = dict(
    type='Few_shot_CocoDetection_RGBT',
    root='../datasets/FLIR/',
    ann_file='Coco_annotations/Val_Annotations.json',
    data_prefix=dict(img_rgb='JPEGImages/', img_ir='JPEGImages/'),
    transforms=[
        dict(type='LoadMSImagesFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='MSFixScaleResize', scale=(640, 640), keep_ratio=False),
        dict(
            type='MSPackDetInputs',
            meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape',
                       'img_shape', 'scale_factor', 'flip', 'flip_direction',
                       'text', 'custom_entities'))
    ],
    support_transforms=[
        dict(type='LoadMSImagesFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='MSHorizontalFlip', prob=0.5),
        dict(type='MSResize', scale=(640, 640), keep_ratio=False),
        dict(
            type='MSPackDetInputs',
            meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape',
                       'img_shape', 'scale_factor', 'flip', 'flip_direction'))
    ],
    stage='meta_learning',
    n_way=5,
    num_shots=1,
    min_area_support=256,
    key_rgb='file_name_RGB',
    key_ir='file_name_IR',
    meta_info=dict(
        ALL_CLASSES=('bicycle', 'car', 'person'),
        BASE_CLASSES=('bicycle', 'car'),
        NOVEL_CLASSES=('person', ),
        mapping=dict(bicycle=1, car=2, person=3),
        palette=[(220, 20, 60), (0, 0, 142), (119, 11, 32)]),
    test_mode=True)
support_test_pipeline = [
    dict(type='LoadMSImagesFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MSResize', scale=(640, 640), keep_ratio=False),
    dict(
        type='NormalizeImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]),
    dict(
        type='MSPackDetInputs',
        meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape',
                   'img_shape', 'scale_factor', 'flip', 'flip_direction'))
]
support_data = dict(
    type='Support_RGBT',
    key_rgb='file_name_RGB',
    key_ir='file_name_IR',
    root='../datasets/FLIR/',
    meta_info=dict(
        ALL_CLASSES=('bicycle', 'car', 'person'),
        BASE_CLASSES=('bicycle', 'car'),
        NOVEL_CLASSES=('person', ),
        mapping=dict(bicycle=1, car=2, person=3),
        palette=[(220, 20, 60), (0, 0, 142), (119, 11, 32)]),
    ann_file='few_shot/seed2/5.json',
    data_prefix=dict(img_rgb='JPEGImages/', img_ir='JPEGImages/'),
    transforms=[
        dict(type='LoadMSImagesFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='MSResize', scale=(640, 640), keep_ratio=False),
        dict(
            type='NormalizeImage',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375]),
        dict(
            type='MSPackDetInputs',
            meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape',
                       'img_shape', 'scale_factor', 'flip', 'flip_direction'))
    ],
    min_area_support=256,
    stage='few_shot_finetune')
metric = dict(
    type='CocoMetric',
    classwise=True,
    dataset_meta=dict(
        ALL_CLASSES=('bicycle', 'car', 'person'),
        BASE_CLASSES=('bicycle', 'car'),
        NOVEL_CLASSES=('person', ),
        mapping=dict(bicycle=1, car=2, person=3),
        palette=[(220, 20, 60), (0, 0, 142), (119, 11, 32)]),
    proposal_nums=(100, 1, 10),
    stage='few_shot_finetune',
    ann_file='../datasets/FLIR/Coco_annotations/Val_Annotations.json',
    metric='bbox')
custom_callbacks = [
    dict(
        type='TrainableParamsHook',
        stage='few_shot_finetune',
        metainfo=dict(
            ALL_CLASSES=('bicycle', 'car', 'person'),
            BASE_CLASSES=('bicycle', 'car'),
            NOVEL_CLASSES=('person', ),
            mapping=dict(bicycle=1, car=2, person=3),
            palette=[(220, 20, 60), (0, 0, 142), (119, 11, 32)]),
        Ignore_params=['backbone', 'reference_points', 'sampling_offsets']),
    dict(type='PrototypeCallback', episode_size=1, max_iters=100),
    dict(
        type='SaveFewShotMetricsCallback',
        monitor='coco/novel_map_50',
        rule='greater',
        summary_file='flir_fewshot.csv'),
    dict(type='LoggingCallback', log_every_n_steps=5),
    dict(
        type='VizualizationCallback',
        save_dir='visualization',
        palette=[(220, 20, 60), (0, 0, 142), (119, 11, 32)],
        min_score_threshold=0.35,
        interval=10)
]
monitor_metric = 'coco/novel_map_50'
train_bs = 2
val_bs = 1
num_workers = 2
epochs = 100
save_checkpoint_interval = 10
freq = 5
clip_max_norm = 0.1
batch_transform = [
    dict(
        type='MSBatchRandomResize',
        scales=[(480, 480), (512, 512), (544, 544), (640, 640), (640, 640),
                (640, 640), (704, 704), (768, 768), (768, 768), (832, 832),
                (896, 896), (896, 896)])
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-05, weight_decay=0.0001))
param_scheduler = [
    dict(type='LinearLR', total_iters=10, start_factor=0.01, by_epoch=True)
]
