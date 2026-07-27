# config file for testing the m3fd dataset
_base_ = ['../detection/fsmodnet.py',]
eval_size = (1024, 1024)
eval_size_support = (1024, 1024)

metainfo = {
    'ALL_CLASSES': ("car", "truck", "pickup","tractor","camper", "boat", "van"),
    'BASE_CLASSES': ("car", "truck", "pickup", "camper"),
    'NOVEL_CLASSES': ("tractor", "boat", "van"),
    'mapping': {"car":1, "truck":2, "pickup":3,"tractor":4,"camper":5, "boat":6, "van":7},
    'palette': [
        (220, 20, 60),
        (0, 0, 142),
        (119, 11, 32),
        (0, 60, 100),
        (0, 0, 230),
        (106, 0, 228),
    ]
}


root = '../datasets/VEDAI/'
num_episodes = 1
num_classes = len(metainfo['ALL_CLASSES'])  

model = dict(
    num_episodes=num_episodes,
    num_classes=num_classes,
    criterion=dict(
        type='DINOCriterion',
        num_classes=num_classes,
    ),
    class_embed=dict(
        type='ContrastiveEmbed',
        max_text_len=num_classes,
    ),
)


train_pipeline = [
    dict(type='LoadMSImagesFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MSPhotoMetricDistortion', prob=0.5),
    #dict(type='MSExpand', mean=_base_.mean, prob=0.5, ratio_range=(1, 4)),
    dict(type='MSRandomIoUCrop', prob=0.8),
    dict(type='FilterAnnotations', min_wh=(1e-2,1e-2)),
    dict(
        type='MSResize',
        scale=eval_size,
        keep_ratio=False),
    dict(type='MSHorizontalFlip', prob=0.5),
    dict(
        type='MSPackDetInputs',
        meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape', 'img_shape', 'text',
                   'scale_factor', 'flip', 'flip_direction'))
]

support_pipeline = [
    dict(type='LoadMSImagesFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MSHorizontalFlip', prob=0.5),
    dict(
        type='MSResize',
        scale=eval_size_support,
        keep_ratio=False),
    dict(
        type='MSPackDetInputs',
        meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction'))
]

test_pipeline = [
    dict(type='LoadMSImagesFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MSFixScaleResize',
        scale=eval_size,
        keep_ratio=False,),
    dict(
        type='MSPackDetInputs',
        meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]



train_data = dict(
    type='Few_shot_CocoDetection_RGBT',
    root=root,
    ann_file='Coco_annotations/Train_Annotations.json',
    data_prefix=dict(img_rgb='images/rgb', img_ir='images/ir'),
    transforms=train_pipeline,
    support_transforms=support_pipeline,
    stage= 'meta_learning',
    n_way=7,
    num_shots=1,
    min_area_support=256,
    key_rgb='file_name',
    key_ir='file_name',
    meta_info=metainfo,
    test_mode=False,
    max_refetch=100,
    with_support=False,
)





val_data = dict(
    type='Few_shot_CocoDetection_RGBT',
    root=root,
    ann_file='Coco_annotations/Val_Annotations.json',
    data_prefix=dict(img_rgb='images/rgb', img_ir='images/ir'),
    transforms=test_pipeline,
    support_transforms=support_pipeline,
    stage= 'meta_learning',
    n_way=7,
    num_shots=1,
    min_area_support=256,
    key_rgb='file_name',
    key_ir='file_name',
    meta_info=metainfo,
    test_mode=True,
    with_support=False,
)
test_data = val_data



support_test_pipeline = [
    dict(type='LoadMSImagesFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MSResize',
        scale=eval_size_support,
        keep_ratio=False),
    dict(type='NormalizeImage',
         mean=_base_.mean,
        std=_base_.std,
        ),
    dict(
        type='MSPackDetInputs',
        meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction'))
]


support_data = dict(
        type='Support_RGBT',
        key_rgb='file_name',
        key_ir='file_name',
        root=root,
        meta_info=metainfo,
        ann_file='Coco_annotations/Train_Annotations.json',
        data_prefix=dict(img_rgb='images/rgb', img_ir='images/ir'),
        transforms=support_test_pipeline,
        min_area_support=256,
        stage='meta_learning',
        )

metric = dict(
    type='CocoMetric',
    classwise=True,
    dataset_meta=metainfo,
    proposal_nums=(100, 1, 10),
    stage='meta_learning',
    ann_file=root + 'Coco_annotations/Val_Annotations.json',
    metric='bbox',
)


custom_callbacks = [
    dict(type='TrainableParamsHook',Ignore_params=['language_model'],),
    dict(type='EMACallback', momentum=0.0001, ema_type='exponential', gamma=2000),
    #dict(type='PrototypeCallback', episode_size=episode_size, max_iters=100),
    dict(type='LoggingCallback'),
    dict(type='VizualizationCallback', palette=metainfo['palette'], min_score_threshold=0.25, interval=10),
]
monitor_metric = 'coco/base_map_50'


train_bs = 4
val_bs = 1
num_workers = 4
epochs = 50
save_checkpoint_interval = 5
freq = 100
clip_max_norm = 0.1
batch_transform = [
    dict(
    type='MSBatchRandomResize',
    scales=[ (896, 896), ( 1024, 1024)],
)
]


optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
         lr=2e-4,
         weight_decay=1e-4,
    ),
    paramwise_cfg=dict(custom_keys={
            'backbone': dict(lr_mult=0.05),
            'backbone_ir': dict(lr_mult=0.05),
        }
    )
)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=epochs,
        by_epoch=True,
        milestones=[45],
        gamma=0.1)
]

