import os
import numpy as np
import cv2
from pycocotools.coco import COCO

from utils.image_draw import draw_rectangle_with_name


def trans_coco_dataset(coco_dataset_path):
    label_path = os.path.join(coco_dataset_path, 'labels')
    if not os.path.exists(label_path):
        os.mkdir(label_path)

    for fn in ['train', 'val']:
        instance_file_path = os.path.join(coco_dataset_path, 'annotations/instances_{}2017.json'.format(fn))
        coco = COCO(instance_file_path)

        # save class names
        cls_ids = coco.getCatIds()
        categories = coco.loadCats(cls_ids)
        cls_names = [cat['name'] for cat in categories]
        with open(os.path.join(coco_dataset_path, 'classes'), 'w') as class_writer:
            for cls_name in cls_names:
                class_writer.write(cls_name + '\n')

        # save image annotations
        with open(os.path.join(coco_dataset_path, fn), 'w') as list_file_writer:
            i = 0
            img_ids = coco.getImgIds()
            total_imgs = len(img_ids)
            for img_id in img_ids:
                img_info = coco.loadImgs(img_id)[0]
                ann_id = coco.getAnnIds(imgIds=img_id)
                ann_info = coco.loadAnns(ann_id)

                file_name = img_info['file_name'].split('.')[0]
                list_file_writer.write(file_name + '\n')

                with open(os.path.join(label_path, file_name), 'w') as ann_writer:
                    for ann in ann_info:
                        bbox = _xywh2xxyy(ann['bbox'])
                        cat_id = cls_ids.index(ann['category_id']) + 1
                        ann_writer.write(' '.join([str(bb) for bb in bbox] + [str(cat_id)]) + '\n')

                i += 1
                if i % 1000 == 0:
                    print('Processed {} image: {} {}/{}'.format(fn, file_name, str(i), str(total_imgs)))

    print('Done!')


def _xywh2xxyy(data):
    x, y, w, h = data
    x1 = round(x)
    y1 = round(y)
    x2 = round(x + w)
    y2 = round(y + h)
    return [x1, y1, x2, y2]


def load_translated_data(coco_dataset_path):
    train_file = os.path.join(coco_dataset_path, 'train')
    val_file = os.path.join(coco_dataset_path, 'val')
    cls_file = os.path.join(coco_dataset_path, 'classes')

    with open(train_file, 'r') as reader:
        train_file_list = [line[:-1] for line in reader]
        # print(train_file_list[0])

    with open(val_file, 'r') as reader:
        val_file_list = [line[:-1] for line in reader]
        # print(val_file_list[0])

    with open(cls_file, 'r') as reader:
        cls_list = [line[:-1] for line in reader]

    return train_file_list, val_file_list, ['BG'] + cls_list


def get_label_infos(label_file_path):
    with open(label_file_path, 'r') as reader:
        bboxes = []
        categories = []
        for line in reader:
            x1, y1, x2, y2, cls = line.split()
            bboxes.append([int(x1), int(y1), int(x2), int(y2)])
            categories.append(int(cls))
    return bboxes, categories


if __name__ == '__main__':
    COCO_PATH = '/media/wx/新加卷/datasets/COCODataset'

    # coco = COCO(os.path.join(COCO_PATH, 'annotations/instances_train2017.json'))
    # ann_id = coco.getAnnIds(imgIds=[522418])
    # ann = coco.loadAnns(ann_id)
    # cls_nanmes = [coco.loadCats(a['category_id']) for a in ann]
    # # print(cls_nanmes)
    #
    # cls_ids = coco.getCatIds()
    # clss = coco.loadCats(cls_ids)
    # print(cls_ids)
    # print(len(clss))
    # print(clss)

    # trans_coco_dataset(COCO_PATH)

    train_file_list, val_file_list, cls_names = load_translated_data(COCO_PATH)

    for fn in train_file_list:
        img_file_path = os.path.join(COCO_PATH, 'train2017', fn + '.jpg')
        label_file_path = os.path.join(COCO_PATH, 'labels', fn)

        img = cv2.imread(img_file_path)
        bboxes, categories = get_label_infos(label_file_path)
        # print(categories)

        img = draw_rectangle_with_name(img, bboxes, categories, cls_names)

        cv2.imshow('coco', img)
        if cv2.waitKey(2000) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

