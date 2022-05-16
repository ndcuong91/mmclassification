# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmcls.apis import inference_model, init_model, show_result_pyplot

config_file ='/home/cuongnd/PycharmProjects/mmclassification/configs/resnet/resnet50_8xb16_doc_quality.py'
img_path ='/data_backup/cuongnd/PVCombank/pvcom/uploads_pvcombank/20220407/103.143.207.84/70f96c92-ec53-47a2-840f-8bf8c6b58ace/023459_giay_dkmst_images.jpg'
ckpt='/home/cuongnd/PycharmProjects/mmclassification/tools/work_dirs/resnet50_8xb16_doc_quality/best_accuracy_epoch_113.pth'
def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file', default = img_path)
    parser.add_argument('--config', help='Config file', default = config_file)
    parser.add_argument('--checkpoint', help='Checkpoint file', default = ckpt)
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_model(model, args.img)
    # show the results
    show_result_pyplot(model, args.img, result)


if __name__ == '__main__':
    main()
