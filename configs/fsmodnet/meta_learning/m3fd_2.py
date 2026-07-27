# config file for testing the flir dataset
_base_ = ['m3fd_1.py',]
metainfo = {
    'ALL_CLASSES': ('People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck'),
    'BASE_CLASSES': ('People', 'Car', 'Lamp', 'Truck'),
    'NOVEL_CLASSES': ('Bus', 'Motorcycle'),
    'mapping': {'People':0, 'Car':1, 'Bus':2, 'Motorcycle':3, 'Lamp':4, 'Truck':5},
    'palette': [
        (220, 20, 60),
        (0, 0, 142),
        (119, 11, 32),
        (0, 60, 100),
        (0, 0, 230),
        (106, 0, 228),
    ]
}


train_data = dict(meta_info=metainfo)
val_data = dict( meta_info=metainfo)
support_data = dict(meta_info=metainfo)

metric = dict(dataset_meta=metainfo)

custom_callbacks = [
    dict(type='TrainableParamsHook',Ignore_params=[],),
    dict(type='PrototypeCallback', episode_size=_base_.episode_size, max_iters=100),
    dict(type='LoggingCallback'),
    
]

monitor_metric = 'coco/base_map_50'
