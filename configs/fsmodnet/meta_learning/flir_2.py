# config file for testing the flir dataset
_base_ = ['flir_1.py',]
metainfo = {
    'ALL_CLASSES': ('bicycle', 'car', 'person'),
    'BASE_CLASSES': ('car', 'person'),
    'NOVEL_CLASSES': ('bicycle',),
    'mapping': { 'bicycle':1, 'car':2, 'person':3},
    'palette': [
        (220, 20, 60),
        (0, 0, 142),
        (119, 11, 32)
    ]
}


train_data = dict(meta_info=metainfo)
val_data = dict( meta_info=metainfo)
test_data = None
support_data = dict(meta_info=metainfo)

metric = dict(dataset_meta=metainfo)

custom_callbacks = [
    dict(type='TrainableParamsHook',Ignore_params=[],),
    dict(type='PrototypeCallback', episode_size=_base_.episode_size, max_iters=100),
    dict(type='LoggingCallback'),
    
]

monitor_metric = 'coco/base_map_50'
