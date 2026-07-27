# config file for testing the flir dataset
_base_ = ['../../detection/fsmodnet.py',]
eval_size = (640, 640)
mean=[123.675, 116.28, 103.53]
std=[58.395, 57.12, 57.375]

metainfo = {
    'ALL_CLASSES': ('bicycle', 'car', 'person'),
    'BASE_CLASSES': ('bicycle', 'car'),
    'NOVEL_CLASSES': ('person',),
    'mapping': { 'bicycle':1, 'car':2, 'person':3},
    'palette': [
        (220, 20, 60),
        (0, 0, 142),
        (119, 11, 32)
    ]
}


root = '../datasets/FLIR/'
episode_size = 1
num_episodes = 2
num_classes = 3

model = dict(
    backbone=dict(
        _delete_=True,
        type='DINOModel',
        model='dinov2_vits14_reg',
        pretrained="/share/projects/fsmod/coco_pretrained_weights/dinov2_vits14_reg4_pretrain.pth",
        n=1,
        reshape=True,
        return_class_token=False,
        feature_levels=[1, 2, 3],
    ),
    neck=dict(
         in_channels=[192, 384, 384],
    ),
    task_enconding=dict(max_len=episode_size),
    episode_size=episode_size,
    num_episodes=num_episodes,
    num_classes=num_classes,
    meta_score=dict(
            out_features=num_classes,
        ),
    criterion=dict(
        num_classes=num_classes,
    ),
)

train_pipeline = [
    dict(type='LoadMSImagesFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MSPhotoMetricDistortion', prob=0.5),
    dict(type='MSExpand', mean=mean, prob=0.5, ratio_range=(1, 4)),
    dict(type='MSRandomIoUCrop', prob=0.8),
    dict(type='FilterAnnotations', min_wh=(1e-2,1e-2)),
    dict(
        type='MSResize',
        scale=eval_size,
        keep_ratio=False),
    dict(type='MSHorizontalFlip', prob=0.5),
    dict(
        type='MSPackDetInputs',
        meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction'))
]

support_pipeline = [
    dict(type='LoadMSImagesFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MSHorizontalFlip', prob=0.5),
    dict(
        type='MSResize',
        scale=eval_size,
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
    data_prefix=dict(img_rgb='JPEGImages/', img_ir='JPEGImages/'),
    transforms=train_pipeline,
    support_transforms=support_pipeline,
    stage= 'meta_learning',
    n_way=5,
    num_shots=1,
    min_area_support=256,
    key_rgb='file_name_RGB',
    key_ir='file_name_IR',
    meta_info=metainfo,
    test_mode=False,
    max_refetch=100,
)





val_data = dict(
    type='Few_shot_CocoDetection_RGBT',
    root=root,
    ann_file='Coco_annotations/Val_Annotations.json',
    data_prefix=dict(img_rgb='JPEGImages/', img_ir='JPEGImages/'),
    transforms=test_pipeline,
    support_transforms=support_pipeline,
    stage= 'meta_learning',
    n_way=5,
    num_shots=1,
    min_area_support=256,
    key_rgb='file_name_RGB',
    key_ir='file_name_IR',
    meta_info=metainfo,
    test_mode=True,
)
test_data = val_data



support_test_pipeline = [
    dict(type='LoadMSImagesFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MSResize',
        scale=eval_size,
        keep_ratio=False),
    dict(type='NormalizeImage',
         mean=mean,
        std=std,
        ),
    dict(
        type='MSPackDetInputs',
        meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction'))
]


support_data = dict(
        type='Support_RGBT',
        key_rgb='file_name_RGB',
        key_ir='file_name_IR',
        root=root,
        meta_info=metainfo,
        ann_file='Coco_annotations/Train_Annotations.json',
        data_prefix=dict(img_rgb='JPEGImages/', img_ir='JPEGImages/'),
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
    dict(type='TrainableParamsHook',Ignore_params=[],),
    dict(type='PrototypeCallback', episode_size=episode_size, max_iters=100),
    dict(type='LoggingCallback'),
    dict(type='VizualizationCallback', save_dir='visualization', palette=metainfo['palette'], min_score_threshold=0.35, interval=30),
    
]

monitor_metric = 'coco/base_map_50'


train_bs = 4
val_bs = 1
num_workers = 4
epochs = 50
save_checkpoint_interval = 10
freq = 1
clip_max_norm = 0.1
batch_transform = [
    dict(
    type='MSBatchRandomResize',
    scales=[        (480, 480),( 512, 512),( 544, 544),
                    (640, 640),( 640, 640),( 640, 640),
                    (704, 704),( 768, 768),( 768, 768),
                    (832, 832),( 896, 896),( 896, 896),],
)
]


optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
         lr=1e-4,
         weight_decay=1e-4,
    ),
    paramwise_cfg=dict(custom_keys={
            'backbone': dict(lr_mult=0.1),
            'backbone_ir': dict(lr_mult=0.1),
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

