# config file for testing the flir dataset
_base_ = ['../detection/dab_upn.py',]
eval_size = (640, 640)


metainfo = {
    'ALL_CLASSES': (
        'person','bicycle','car','motorcycle','airplane','bus','train',
        'truck','boat','traffic light','fire hydrant','stop sign',
        'parking meter','bench','bird','cat','dog','horse','sheep', 'fire hydrant','stop sign','parking meter','bench',
        'elephant','bear','zebra','giraffe',
        'backpack','umbrella','handbag','tie','suitcase',
        'frisbee','skis','snowboard','sports ball','kite',
        'baseball bat','baseball glove','skateboard','surfboard','tennis racket',
        'wine glass','cup','fork','knife','spoon','bowl',
        'banana','apple','sandwich','orange','broccoli','carrot',
        'hot dog','pizza','donut','cake',
        'bed','toilet','tv',
        'laptop','mouse','remote','keyboard','cell phone',
        'microwave','oven','toaster','sink','refrigerator',
        'book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
    ),
     'BASE_CLASSES': (
        'fire hydrant','stop sign','parking meter','bench',
        'elephant','bear','zebra','giraffe',
        'backpack','umbrella','handbag','tie','suitcase',
        'frisbee','skis','snowboard','sports ball','kite',
        'baseball bat','baseball glove','skateboard','surfboard','tennis racket',
        'wine glass','cup','fork','knife','spoon','bowl',
        'banana','apple','sandwich','orange','broccoli','carrot',
        'hot dog','pizza','donut','cake',
        'bed','toilet','tv',
        'laptop','mouse','remote','keyboard','cell phone',
        'microwave','oven','toaster','sink','refrigerator',
        'book','clock','vase','scissors','teddy bear','hair drier','toothbrush',
        'bird','cat','dog','horse','sheep','cow',
        'bottle',
        'chair','couch','potted plant',
        'dining table','boat', 'airplane',
    ),

    'NOVEL_CLASSES': (
        'person','bicycle','car','motorcycle','bus','train', 'truck',
        'traffic light',
        
    ),
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


root = '../datasets/VTUAV/VTUAV_train/'


model = dict(
    type='DABDETR_UPN',
    use_upn=True
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


coco_data = dict(
    type='CocoDetection_RGBT',
    root='../datasets/COCO/',
    ann_file='Coco_annotations/instances_train2017.json',
    data_prefix=dict(img_rgb='JPEGImages/train2017/', img_ir='JPEGImages/train2017/'),
    transforms=train_pipeline,
    key_rgb='file_name',
    key_ir='file_name',
    meta_info=metainfo,
    test_mode=False,
    max_refetch=100,
    stage='few_shot_finetune',
    with_support=False,
)


rgbt_data = dict(
         type='CocoDetection_RGBT',
        root=root,
        ann_file='train.json',
        transforms=train_pipeline,
        key_rgb='file_name',
        key_ir='file_name',
        meta_info=metainfo,
        test_mode=False,
        max_refetch=100,
        with_support=False,
        data_prefix=dict(img_rgb='rgb/', img_ir='ir/'),
        stage='few_shot_finetune',
        )


train_data = dict(type='ConcatDataset',
            datasets= [rgbt_data], # repeat the rgbt_data to balance the dataset and make the model focus more on the target dataset
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
steps = 4000000      
save_checkpoint_interval = 5000
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


