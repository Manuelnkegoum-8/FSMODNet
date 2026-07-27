# config file for testing the flir dataset
_base_ = ['mfad_1.py',]
metainfo = {
    'ALL_CLASSES': ('car', 'bus', 'truck', 'pedestrian', 'ebikerider', 'cyclist'),
    'BASE_CLASSES': ('car', 'pedestrian', 'ebikerider', 'cyclist'),
    'NOVEL_CLASSES': ('bus', 'truck',),
    'mapping': {'car':0, 'bus':1, 'truck':2, 'pedestrian':3, 'ebikerider':4, 'cyclist':5},
    'palette': [
        (220, 20, 60),
        (0, 0, 142),
        (119, 11, 32),
        (0, 60, 100),
        (0, 0, 230),
    ]
}


root = '../datasets/MFAD/'
episode_size = 2
num_episodes = 4
num_classes = 6

model = dict(
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
