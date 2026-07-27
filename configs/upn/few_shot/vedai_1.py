_base_ = [ '../meta_learning/vedai_1.py',]



train_data = dict(
        ann_file='few_shot/seed2/5.json',
        stage='few_shot_finetune',
        )


val_data = dict(stage='few_shot_finetune',)


support_data = dict(ann_file='few_shot/seed2/5.json',
                    stage='few_shot_finetune',
                 )




#''reference_points', 'sampling_offsets''
custom_callbacks = [
        dict(
        type='TrainableParamsHook', metainfo=_base_.metainfo,
            Ignore_params=['backbone', 'language_model'],),
        dict(type='EMACallback', momentum=0.0001, ema_type='exponential', gamma=2000),
        dict(type='SaveFewShotMetricsCallback', monitor='coco/novel_map_50', rule='greater', summary_file='vedai_fewshot.csv'),
        dict(type='LoggingCallback', log_every_n_steps=5),
         dict(type='VizualizationCallback', palette=_base_.metainfo['palette'], min_score_threshold=0.05, interval=10),
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
        start_factor=0.1,
        by_epoch=True,
    ),
]


metric = dict(
    stage='few_shot_finetune',)



