_base_ = [
    '../_base_/models/efficientnet_b3_finetune.py',
    '../_base_/datasets/doc_crop.py',
    '../_base_/schedules/schedule_200.py',
    '../_base_/default_runtime.py',
]