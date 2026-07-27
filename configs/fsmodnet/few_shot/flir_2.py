_base_ = [ '../meta_learning/flir_2.py',]

num_episodes = 3

model = dict(
    num_episodes=num_episodes,
)


train_data = dict(
        type='Few_shot_CocoDetection_RGBT',
        ann_file='few_shot/seed2/5.json',
        stage='few_shot_finetune',
        with_support=True,
        )


val_data = dict(
        type='Few_shot_CocoDetection_RGBT',
        ann_file='Coco_annotations/Val_Annotations.json',
        stage='few_shot_finetune',
        with_support=False,
        )


support_data = dict(
        type='Support_RGBT',
        ann_file='few_shot/seed2/5.json',
        stage='few_shot_finetune',
        )




#''reference_points', 'sampling_offsets''
custom_callbacks = [
        dict(
        type='TrainableParamsHook', stage='few_shot_finetune', metainfo=_base_.metainfo,
            Ignore_params=['backbone', 'reference_points','sampling_offsets'],),
         dict(type='PrototypeCallback', episode_size=_base_.episode_size, max_iters=100),
        dict(type='SaveFewShotMetricsCallback', monitor='coco/novel_map_50', rule='greater', summary_file='flir_fewshot.csv'),
        dict(type='LoggingCallback', log_every_n_steps=5),
    ]

monitor_metric = 'coco/novel_map_50'

train_bs = 2
val_bs = 1
num_workers = 2
epochs = 100
save_checkpoint_interval = 10
freq = 5

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00005, 
        weight_decay=0.0001),
) 



param_scheduler = [
    dict(
        type='LinearLR',
        total_iters=10,
        start_factor=0.01,
        by_epoch=True,
    ),
]


metric = dict(
    stage='few_shot_finetune',)



