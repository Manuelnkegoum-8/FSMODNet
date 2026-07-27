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

root = '../datasets/VTUAV/VTUAV_train/'




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
        meta_keys=('img_id', 'img_path_rgb', 'img_path_ir', 'ori_shape', 'img_shape', 'text',
                   'scale_factor', 'flip', 'flip_direction'))
]





train_data = dict(
        type='MSImageListDataset',
        data_root=root,
        data_prefix=dict(img_rgb='rgb/', img_ir='ir/'),
        pipeline=train_pipeline,
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
steps = 40000       
save_checkpoint_interval = 1000
freq = 100
clip_max_norm = 0.1
batch_transform = [
    dict(
    type='MSBatchRandomResize',
    scales=[ 
                    (640, 640),( 640, 640),( 640, 640),
                    (704, 704),( 768, 768),( 768, 768),
                    (832, 832),( 896, 896),( 896, 896),
                    (960, 960),(1024, 1024),(1024, 1024),
                    (1088, 1088),(1152, 1152),(1152, 1152),
                    (1216, 1216),(1280, 1280),(1280, 1280),],
)
]


optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
         lr=1e-4,
         weight_decay=1e-4,
    ),
    paramwise_cfg=dict(custom_keys={
            'backbone': dict(lr_mult=0.),
            'backbone_ir': dict(lr_mult=0.),
        }
    )
)

param_scheduler = None


