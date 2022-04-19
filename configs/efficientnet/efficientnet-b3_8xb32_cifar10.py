_base_ = [
    '../_base_/models/efficientnet_b3_finetune.py',
    '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/schedule_200.py',
    '../_base_/default_runtime.py',
]