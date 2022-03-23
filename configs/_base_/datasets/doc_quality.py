# dataset settings
dataset_type = 'DocQuality'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop',
        size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='CenterCrop',
        crop_size=224,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='/home/cuongnd/PycharmProjects/mmclassification/data/doc_quality/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/home/cuongnd/PycharmProjects/mmclassification/data/doc_quality/val',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='/home/cuongnd/PycharmProjects/mmclassification/data/doc_quality/val',
        pipeline=test_pipeline,
        test_mode=True))