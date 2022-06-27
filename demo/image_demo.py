# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import time
from torchvision import transforms
from mmcls.apis import inference_model, init_model
from torch import jit
from PIL import Image
import torch
net = jit.load('/home/cuongnd/PycharmProjects/mmclassification/tools/deployment/tmp.pt')

config_file ='/home/cuongnd/PycharmProjects/mmclassification/tools/work_dirs/efficientnet-b3_8xb32_doc_quality_2022-06-16_16-09/efficientnet-b3_8xb32_doc_quality.py'
img_path ='/home/cuongnd/Documents/GTTT/CCCD_1b0ae220ecdf4ac3ba9aab2e19d2079e_roi.jpg'
ckpt = '/home/cuongnd/PycharmProjects/mmclassification/tools/work_dirs/efficientnet-b3_8xb32_doc_crop_2022-05-21_21-07/epoch_116_907_input_256_256.pth'
def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file', default = img_path)
    parser.add_argument('--config', help='Config file', default = config_file)
    parser.add_argument('--checkpoint', help='Checkpoint file', default = ckpt)
    parser.add_argument('--device', default='cpu', help='Device used for inference')
    args = parser.parse_args()

    x = torch.ones(1, 3, 256, 256)

    img = Image.open(img_path)
    conv_img = transforms.ToTensor(img)

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    begin = time.time()
    for i in range(100):
        print(net(x))
    result = inference_model(model, args.img)
    #print(result)
    end = time.time()
    print('time:',1000*(end-begin))
    # show the results
    # show_result_pyplot(model, args.img, result)


if __name__ == '__main__':
    main()
