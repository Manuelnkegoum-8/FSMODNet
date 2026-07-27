# config file for testing the flir dataset
_base_ = ['../detection/fsmodnet_upn.py',]
eval_size = (640, 640)


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


model = dict(
    type='DABDETR_UPN',
    use_upn=True,
)

train_pipeline = [
    dict(type='LoadMSImagesFromFile'),
    dict(type='LoadAnnotations', with_bbox=False, with_label=False),
    dict(type='MSPhotoMetricDistortion', prob=0.5),
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



rgbt_data = dict(
    type='CocoDetection_RGBT',
    root=root,
    ann_file='Coco_annotations/Train_Annotations.json',
    data_prefix=dict(img_rgb='JPEGImages/', img_ir='JPEGImages/'),
    transforms=train_pipeline,
    stage= 'meta_learning',
    key_rgb='file_name_RGB',
    key_ir='file_name_IR',
    meta_info=metainfo,
    with_support=False,
)


train_data = dict(type='ConcatDataset',
            datasets= [rgbt_data,],
)

val_data =  None
test_data = val_data



custom_callbacks = [
    dict(type='TrainableParamsHook',Ignore_params=['language_model','backbone'],),
    dict(type='EMACallback', momentum=0.0001, ema_type='exponential', gamma=2000),
    dict(type='LoggingCallback'),
]

monitor_metric = 'coco/base_map_50'


train_bs = 4
val_bs = 1
num_workers = 4
epochs = -1
steps = 20000       
save_checkpoint_interval = 1000
freq = 100
clip_max_norm = 0.1
batch_transform = [
    dict(
    type='MSBatchRandomResize',
    scales=[ 
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
)

param_scheduler = None


