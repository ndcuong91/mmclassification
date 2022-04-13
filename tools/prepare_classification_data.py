import os, shutil


def move_files_to_class_dir(imgs_dir, anno_file, synset_file, output_dir):
    '''
    Phân lớp cho file ảnh dựa vào file anno theo format Imagenet v1.0 của cvat
    :param imgs_dir:
    :param anno_file:
    :param output_dir:
    :return:
    '''
    with open(anno_file, mode='r', encoding='utf-8') as f:
        list_files = f.readlines()

    list_files = [f.replace('\n','').rstrip() for f in list_files]

    with open(synset_file, mode='r', encoding='utf-8') as f:
        list_classes = f.readlines()

    if not os.path.exists(os.path.join(output_dir, 'normal')):
        os.makedirs(os.path.join(output_dir, 'normal'))
    for idx, fi in enumerate(list_files):
        print(idx, fi)
        split_name = fi.split('.') #abc.jpg 0 --> ['abc','jpg 0']
        if len(split_name)>1:
            split_fi = split_name[1].split(' ')
            split_name[0]= split_name[0].replace('"','')
            img_basename = '.'.join([split_name[0],split_fi[0]])
            img_path = os.path.join(imgs_dir, img_basename )
            if len(split_fi)>1:
                cls_idx = split_fi[1]
                cls = list_classes[int(cls_idx)]

                if not os.path.exists(os.path.join(output_dir, cls)):
                    os.makedirs(os.path.join(output_dir, cls))

                dst_path = os.path.join(output_dir, cls, img_basename)
                shutil.copy(img_path, dst_path)
            else:
                shutil.copy(img_path, os.path.join(output_dir, 'normal', fi))
        else:
            print('Error')



if __name__ == '__main__':
    imgs_dir = '/home/cuongnd/PycharmProjects/mmclassification/data/doc_quality/mcocr_hoadon/img'
    anno_file = '/home/cuongnd/PycharmProjects/mmclassification/data/doc_quality/mcocr_hoadon/default.txt'
    synset_file = '/home/cuongnd/PycharmProjects/mmclassification/data/doc_quality/mcocr_hoadon/synsets.txt'
    output_dir = '/home/cuongnd/PycharmProjects/mmclassification/data/doc_quality/mcocr_hoadon/classes'
    move_files_to_class_dir(imgs_dir,
                            anno_file,
                            synset_file,
                            output_dir)