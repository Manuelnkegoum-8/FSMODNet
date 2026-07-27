# config file for testing the flir dataset
_base_ = ['../detection/dab_upn.py',]
eval_size = (1024, 1024)



metainfo2 = {
    'palette': [
        (220,  20,  60), (119,  11,  32), (  0,   0, 142), (  0,   0, 230),
        (106,   0, 228), (  0,  60, 100), (  0,  80, 100), (  0,   0,  70),
        (  0,   0, 192), (250, 170,  30), (100, 170,  30), (220, 220,   0),
        (175, 116, 175), (250,   0,  30), (165,  42,  42), (255,  77, 255),
        (  0, 226, 252), (182, 182, 255), (  0,  82,   0), (120, 166, 157),
        (110, 76,   0),  (174,  57, 255), (199, 100,   0), ( 72,   0, 118),
        (255, 179, 240), (  0, 125,  92), (209,   0, 151), ( 37,  53, 131),
        (200, 165,  25), ( 68,  36,  31), (115,   0, 217), (154, 104,  80),
        ( 50,  60, 120), (122,  71,  77), (134, 205, 116), ( 10, 156,  36),
        (222, 138, 204), (119, 173, 240), (  0, 155,  98), (165, 200, 167),
        (109, 104, 252), (255,  82,   0), (137, 190, 209), (130,  90,  44),
        (255, 219, 119), (  0, 200, 255), (127, 255, 212), (255, 130,  37),
        (255, 196,   0), (255, 255, 128), (190, 190, 128), (  0, 100, 128),
        (128,   0, 128), (  0, 128, 128), (128, 128,   0), (230, 140,  40),
        (100, 200, 100), (200, 100, 150), (150, 200, 200), ( 50, 150, 250),
        (200,  50,  50), ( 50, 200,  50), ( 50,  50, 200), (200, 200,  50),
        (200,  50, 200), ( 50, 200, 200), (150, 100, 200), (200, 150, 100),
        (100, 150, 200), (200, 100,  50), ( 50, 100, 150), (150, 200,  50),
        (100,  50, 200), ( 50, 150, 100), (200,  50, 100), (100, 200,  50),
        ( 50, 100, 200), (150,  50, 100), (180, 120,  80), (120,  80, 180),
    ],
}
bc = (
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'train',
    'truck',
    'boat',
    'bench',
    'bird',
    'horse',
    'sheep',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'handbag',
    'suitcase',
    'frisbee',
    'skis',
    'kite',
    'surfboard',
    'bottle',
    'fork',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'pizza',
    'donut',
    'chair',
    'bed',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'microwave',
    'oven',
    'toaster',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'toothbrush',
)

nc = (
    'airplane',
    'bus',
    'cat',
    'dog',
    'cow',
    'elephant',
    'umbrella',
    'tie',
    'snowboard',
    'skateboard',
    'cup',
    'knife',
    'cake',
    'couch',
    'keyboard',
    'sink',
    'scissors',
)
ac  = (
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'kite',
    'skateboard',
    'surfboard',
    'bottle',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'bed',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'toothbrush',
)

metainfo = {
    'ALL_CLASSES': ac,
    'BASE_CLASSES': bc,
    'NOVEL_CLASSES': nc,
    'palette': metainfo2['palette'],
}

root = '../datasets/COCO/'
num_episodes = 1
num_classes = len(metainfo['BASE_CLASSES'])  

model = dict(
    num_episodes=num_episodes,
    num_classes=80,
    class_aware_matching=True,
    use_upn=False,
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
    dict(type='MSExpand', mean=_base_.mean, prob=0.5, ratio_range=(1, 4)),
    dict(type='MSRandomIoUCrop', prob=0.8),
    dict(type='FilterAnnotations', min_wh=(1e-2,1e-2)),
    dict(
        type='MSResize',
        scale=eval_size,
        keep_ratio=False),
    dict(type='MSHorizontalFlip', prob=0.5),
    dict(type='RandomLoadText'),
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
    type='CocoDetection_RGBT',
    root=root,
    ann_file='Coco_annotations/instances_train2017.json',
    data_prefix=dict(img_rgb='JPEGImages/train2017/', img_ir='JPEGImages/train2017/'),
    key_rgb='file_name',
    key_ir='file_name',
    transforms=train_pipeline,
    support_transforms=support_pipeline,
    stage= 'few_shot_finetune',
    n_way=5,
    num_shots=1,
    min_area_support=256,
    meta_info=metainfo,
    test_mode=False,
    max_refetch=100,
    with_support=False,
)





val_data = dict(
    type='CocoDetection_RGBT',
    root=root,
    ann_file='Coco_annotations/instances_val2017.json',
    data_prefix=dict(img_rgb='JPEGImages/val2017/', img_ir='JPEGImages/val2017/'),
    transforms=test_pipeline,
    support_transforms=support_pipeline,
    stage= 'few_shot_finetune',
    n_way=5,
    num_shots=1,
    min_area_support=256,
    key_rgb='file_name',
    key_ir='file_name',
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
         mean=_base_.mean,
        std=_base_.std,
        ),
    dict(
        type='MSPackDetInputs',
        meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction'))
]


"""support_data = dict(
        type='Support_RGBT',
        key_rgb='file_name',
        key_ir='file_name',
        root=root,
        meta_info=metainfo,
        ann_file='Coco_annotations/instances_train2017.json',
        data_prefix=dict(img_rgb='JPEGImages/train2017/', img_ir='JPEGImages/train2017/'),
        transforms=support_test_pipeline,
        min_area_support=256,
        stage='meta_learning',
        )"""

support_data = None

metric = dict(
    type='CocoMetric',
    classwise=True,
    dataset_meta=metainfo,
    proposal_nums=(100, 1, 10),
    stage='meta_learning',
    ann_file=root + 'Coco_annotations/instances_val2017.json',
    metric='bbox',
)





custom_callbacks = [
    dict(type='TrainableParamsHook',Ignore_params=['language_model', 'backbone',],),
    dict(type='EMACallback', momentum=0.0001, ema_type='exponential', gamma=2000),
    #dict(type='PrototypeCallback', episode_size=episode_size, max_iters=100),
    dict(type='LoggingCallback', log_every_n_steps=100),
    dict(type='VizualizationCallback', palette=metainfo['palette'], min_score_threshold=0.15, interval=10),
]

monitor_metric = 'coco/base_map_50'


train_bs = 4
val_bs = 1
num_workers = 4
epochs = 50
save_checkpoint_interval = 1
freq = 100
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
            #'backbone': dict(lr_mult=0.1),$
            #'backbone_ir': dict(lr_mult=0.1),
            #'encoder': dict(lr_mult=0.1),
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


