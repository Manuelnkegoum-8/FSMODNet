_base_ = [ '../meta_learning/flir_1_v2.py',]


train_data = dict(
        ann_file='few_shot/seed2/5.json',
        stage='few_shot_finetune',
        with_support=False,
        )


val_data = dict(
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
        type='TrainableParamsHook', metainfo=_base_.metainfo, stage='few_shot_finetune',
            Ignore_params=['backbone', 'language_model'],),
            #dict(type='EMACallback', momentum=0.0001, ema_type='exponential', gamma=2000),
        dict(type='SaveFewShotMetricsCallback', monitor='coco/novel_map_50', rule='greater', summary_file='flir_fewshot.csv'),
        dict(type='LoggingCallback', log_every_n_steps=5),
        dict(type='VizualizationCallback', palette=_base_.metainfo['palette'], min_score_threshold=0.25, interval=10),
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



param_scheduler = None


metric = dict(
    stage='few_shot_finetune',)



