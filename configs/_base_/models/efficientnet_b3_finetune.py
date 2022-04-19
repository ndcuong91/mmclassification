# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNet',
                  arch='b3',
                  init_cfg=dict(
                      type='Pretrained',
                      checkpoint='https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32_in1k_20220119-4b4d7487.pth',
                      prefix='backbone',
                  )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))