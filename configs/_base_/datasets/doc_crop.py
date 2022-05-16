# dataset settings
dataset_type = 'DocCrop'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Albu', transforms=[
        dict(type='RandomRotate90',p=0.5),
        dict(type='Blur',p=0.3),
    ]),
    dict(type='ColorJitter', brightness=0.3, contrast=0.3, saturation=0.3),
    dict(type='RandomGrayscale', gray_prob=0.2),
    dict(type='Resize', size=crop_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=crop_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='/home/cuongnd/PycharmProjects/document_quality_dataset/doc_lack_of_corner/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/home/cuongnd/PycharmProjects/document_quality_dataset/doc_lack_of_corner/val',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='/home/cuongnd/PycharmProjects/document_quality_dataset/doc_lack_of_corner/test',
        pipeline=test_pipeline,
        test_mode=True))

evaluation = dict(interval=1, metric='accuracy', save_best='auto')
