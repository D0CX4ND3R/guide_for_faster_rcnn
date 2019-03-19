import os
import numpy as np
import random
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
                height, width = img_info['height'], img_info['width']
                list_file_writer.write(' '.join([file_name, str(height), str(width), '\n']))

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

    train_file_list = []
    train_label_list = []
    train_image_size_list = []
    with open(train_file, 'r') as reader:
        for line in reader:
            file_name, height, width = line.split()
            train_file_list.append(os.path.join(coco_dataset_path, 'train2017', file_name + '.jpg'))
            train_label_list.append(os.path.join(coco_dataset_path, 'labels', file_name))
            train_image_size_list.append([int(height), int(width)])

    val_file_list = []
    val_label_list = []
    val_image_size_list = []
    with open(val_file, 'r') as reader:
        for line in reader:
            file_name, height, width = line.split()
            val_file_list.append(os.path.join(coco_dataset_path, 'val2017', file_name + '.jpg'))
            val_label_list.append(os.path.join(coco_dataset_path, 'labels', file_name))
            val_image_size_list.append([int(height), int(width)])

    with open(cls_file, 'r') as reader:
        cls_list = [line[:-1] for line in reader]

    return train_file_list, train_label_list, train_image_size_list, \
           val_file_list, val_label_list, val_image_size_list, cls_list


def get_label_infos(label_file_path):
    with open(label_file_path, 'r') as reader:
        bboxes = []
        categories = []
        for line in reader:
            x1, y1, x2, y2, cls = line.split()
            bboxes.append([int(x1), int(y1), int(x2), int(y2)])
            categories.append(int(cls))
    return bboxes, categories


def get_gt_infos(label_file_path):
    with open(label_file_path, 'r') as reader:
        gt_bboxes = []
        for line in reader:
            x1, y1, x2, y2, cls = line.split()
            gt_bboxes.append([int(x1), int(y1), int(x2), int(y2), int(cls)])
    return gt_bboxes


def analyse_dataset(image_list, label_list, cls_names, print_info=True):
    total_image_count = len(image_list)
    processed_image_count = 0
    total_target_count = 0
    bins = np.zeros(shape=(len(cls_names), ), dtype=np.float32)
    color_mean = np.zeros(shape=(1, 3), dtype=np.float32)
    for img_file, label_file in zip(image_list, label_list):
        processed_image_count += 1
        img = cv2.imread(img_file)
        if len(img.shape) == 2:
            img = np.dstack([img] * 3)
        m = img.mean(axis=(0, 1))
        color_mean = (color_mean * (processed_image_count - 1) + m) / processed_image_count

        _, cats = get_label_infos(label_file)
        for cat in cats:
            bins[cat-1] += 1
        total_target_count += len(cats)

        if print_info and processed_image_count % 1000 == 0:
            print('Processed image: {} / {}'.format(processed_image_count, total_image_count))
            print('Get targets:', total_target_count)

    return processed_image_count, total_target_count, color_mean, bins


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

    train_file_list, train_label_list, train_image_size_list, \
    val_file_list, val_label_list, val_image_size_list, cls_names = load_translated_data(COCO_PATH)

    image_count, target_count, color_mean, bins = analyse_dataset(train_file_list, train_label_list, cls_names)
    print('Target_count:', target_count)
    print('Color mean:', color_mean)


    # def _image_batch(image_list, label_list, size_list, batch_size=1):
    #     total_samples = len(image_list)
    #     while True:
    #         ind = random.choice(range(total_samples))
    #         img = cv2.imread(image_list[ind])
    #         gt_bboxes = np.array(get_gt_infos(label_list[ind]))
    #         img_size = np.array(size_list[ind])
    #         yield img, gt_bboxes, img_size
    #
    #
    # cls_names = ['BG'] + cls_names
    #
    # g = _image_batch(train_file_list, train_label_list, train_image_size_list)
    # total_samples = len(train_file_list)
    # while cv2.waitKey(2000) & 0xFF != ord('q'):
    #     img, gt_bboxes, img_size = g.__next__()
    #     img = draw_rectangle_with_name(img, gt_bboxes[:, :-1], gt_bboxes[:, -1], cls_names)
    #     cv2.imshow('coco', img)
    #
    #     print('Image height: {} width: {}'.format(img_size[0], img_size[1]))
    # cv2.destroyAllWindows()

    # for img_file_path, label_file_path, img_size in zip(train_file_list, train_label_list, train_image_size_list):
    #     img = cv2.imread(img_file_path)
    #     bboxes, categories = get_label_infos(label_file_path)
    #     print('Image height: {} width: {}'.format(img_size[0], img_size[1]))
    #
    #     img = draw_rectangle_with_name(img, bboxes, categories, cls_names)
    #
    #     cv2.imshow('coco', img)
    #     if cv2.waitKey(2000) & 0xFF == ord('q'):
    #         break
    #
    # cv2.destroyAllWindows()

