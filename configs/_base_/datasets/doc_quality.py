# dataset settings
dataset_type = 'DocQuality'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512,512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(520,-1)),
    dict(type='CenterCrop',
        crop_size=crop_size),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(520,-1)),
    dict(type='CenterCrop',
        crop_size=crop_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,

    train=dict(
        type=dataset_type,
        data_prefix='/home/cuongnd/PycharmProjects/document_quality_dataset/doc_quality/train',
        #data_prefix='/data_backup/tiep/Dataset/Image/FaceAntispoof/Train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/home/cuongnd/PycharmProjects/document_quality_dataset/doc_quality/val',
        #data_prefix='/data_backup/tiep/Dataset/Image/FaceAntispoof/Val',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='/home/cuongnd/PycharmProjects/document_quality_dataset/doc_quality/val',
        #data_prefix='/data_backup/tiep/Dataset/Image/FaceAntispoof/Val',
        pipeline=test_pipeline,
        test_mode=True))

evaluation = dict(interval=1, metric='accuracy', save_best='auto')