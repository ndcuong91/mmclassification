import os

config_file = '/home/cuongnd/PycharmProjects/mmclassification/configs/efficientnet/efficientnet-b3_8xb32_doc_quality.py'
ckpt_resume = '/home/cuongnd/PycharmProjects/mmclassification/tools/work_dirs/efficientnet-b3_8xb32_doc_crop_2022-05-22_09-17/epoch_18.pth'
ckpt_resume = None
ckpt_test = '/home/cuongnd/PycharmProjects/freeform/main_app/main_flow/doc_quality_check/mmclassification/weights/epoch_153_83.6.pth'
pkl_file  = ckpt_test.replace('.pth','.pkl')
input_shape = 256