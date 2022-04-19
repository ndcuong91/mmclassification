# Copyright (c) OpenMMLab. All rights reserved.
import argparse, os
import os.path as osp

import mmcv
from mmcv import DictAction

from mmcls.datasets import build_dataset
from mmcls.models import build_classifier


config_file = '/home/cuongnd/PycharmProjects/mmclassification/configs/resnet/resnet18_8xb16_doc_quality.py'
res_file = '/home/cuongnd/PycharmProjects/mmclassification/tools/best_accuracy_epoch_176_res.pkl'
out_dir = '/home/cuongnd/PycharmProjects/document_quality_dataset/doc_quality/res/'+os.path.basename(res_file).split('.')[0]
if not os.path.exists(out_dir): os.makedirs(out_dir)
split_success_fail = True
threshold = 0.0

def parse_args():
    parser = argparse.ArgumentParser(description='MMCls evaluate prediction success/fail')
    parser.add_argument('--config', help='test config file path', default = config_file)
    parser.add_argument('--result', help='test result json/pkl file', default = res_file)
    parser.add_argument('--out-dir', help='dir to store output files', default = out_dir)
    parser.add_argument(
        '--topk',
        default=1000,
        type=int,
        help='Number of images to select for success/fail')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()

    return args


def save_imgs(result_dir, folder_name, results, model, filter = False):
    full_dir = osp.join(result_dir, folder_name)
    mmcv.mkdir_or_exist(full_dir)
    mmcv.dump(results, osp.join(full_dir, folder_name + '.json'))

    # save imgs
    show_keys = ['pred_score', 'pred_class', 'gt_class']
    for result in results:
        cont_ok = filter and result['gt_label']==3
        if not filter or cont_ok:
            result_show = dict((k, v) for k, v in result.items() if k in show_keys)
            outfile = osp.join(full_dir, osp.basename(result['filename']))
            model.show_result(result['filename'], result_show, out_file=outfile, norm_size=800)


def plot_confusion_matrix(preds, gts, list_class =[]):
    '''
    Draw confusion matrix
    :param preds: list of preds. eg [0,1,2]
    :param gts: list of ground-truth. eg [0,1,2]
    :return:
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import confusion_matrix

    list_class = [cls[:5] for cls in list_class]

    # Get the confusion matrix
    cf_matrix = confusion_matrix(gts, preds)
    cf_matrix_norm = []
    for line in cf_matrix:
        total = np.sum(line)
        new_line =[]
        for idx, val in enumerate(line):
            new_line.append( round(100*val/total,2))
        cf_matrix_norm.append(new_line)

    print(cf_matrix_norm)
    ax = sns.heatmap(cf_matrix_norm, annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix with labels\n');
    ax.set_xlabel('Predicted ')
    ax.set_ylabel('Ground truth');

    ax.xaxis.set_ticklabels(list_class)
    ax.yaxis.set_ticklabels(list_class)

    ## Display the visualization of the Confusion Matrix.
    # plt.show()
    plt.savefig(os.path.join(out_dir, 'conf_matrix_{}.png'.format(threshold)))


def main():
    args = parse_args()

    # load test results
    outputs = mmcv.load(args.result)
    assert ('pred_score' in outputs and 'pred_class' in outputs
            and 'pred_label' in outputs), \
        'No "pred_label", "pred_score" or "pred_class" in result file, ' \
        'please set "--out-items" in test.py'

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = build_classifier(cfg.model)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    filenames = list()
    for info in dataset.data_infos:
        if info['img_prefix'] is not None:
            filename = osp.join(info['img_prefix'],
                                info['img_info']['filename'])
        else:
            filename = info['img_info']['filename']
        filenames.append(filename)
    gt_labels = list(dataset.get_gt_labels())
    gt_classes = [dataset.CLASSES[x] for x in gt_labels]

    outputs['filename'] = filenames
    outputs['gt_label'] = gt_labels
    outputs['gt_class'] = gt_classes

    need_keys = [
        'filename', 'gt_label', 'gt_class', 'pred_score', 'pred_label',
        'pred_class'
    ]
    outputs = {k: v for k, v in outputs.items() if k in need_keys}
    outputs_list = list()
    for i in range(len(gt_labels)):
        output = dict()
        for k in outputs.keys():
            output[k] = outputs[k][i]
        outputs_list.append(output)

    # sort result
    outputs_list = sorted(outputs_list, key=lambda x: x['pred_score'])

    success = list()
    fail = list()

    list_preds =[]
    list_gts =[]
    total_samples ={'normal':0,'abnormal':0}
    true_samples ={'normal':0,'abnormal':0}
    for output in outputs_list:
        if output['pred_score']>threshold and output['gt_label']!=0:
            list_preds.append(output['pred_label'])
            list_gts.append(output['gt_label'])
            if output['gt_label'] ==3:
                total_samples['normal']+=1
            else :
                total_samples['abnormal']+=1

            if output['pred_label'] == output['gt_label'] and output['pred_label'] ==3:
                true_samples['normal']+=1
            if output['pred_label'] !=3 and output['gt_label'] !=3:
                true_samples['abnormal']+=1

            if output['pred_label'] == output['gt_label']:
                success.append(output)
            else:
                fail.append(output)

    success = success[:args.topk]
    fail = fail[:args.topk]

    print('normal acc:', round(100*true_samples['normal']/total_samples['normal'],2))
    print('abnormal acc:', round(100*true_samples['abnormal']/total_samples['abnormal'],2))

    plot_confusion_matrix(list_preds, list_gts, list_class=dataset.CLASSES)

    if split_success_fail:
        print('Split imgs to success / fail based on pred / gt...')
        #save_imgs(args.out_dir, 'success', success, model)
        save_imgs(args.out_dir, 'fail', fail, model, filter = True)
        print('Done')


if __name__ == '__main__':
    main()

