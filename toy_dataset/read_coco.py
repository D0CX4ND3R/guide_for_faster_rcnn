import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

import json


if __name__ == '__main__':
    ANN_FILE_PATH = '/media/wx/新加卷/datasets/COCODataset/annotations/instances_train2017.json'
    TRAIN_IMAGE_PATH = '/media/wx/新加卷/datasets/COCODataset/train2017'

    coco = COCO(ANN_FILE_PATH)
    catIDs = coco.getCatIds()
    categories = coco.loadCats(catIDs)

    nms = [cat['name'] for cat in categories]
    imgIDs = coco.getImgIds()
    # print(len(imgIDs))

    while cv2.waitKey(2000) & 0xFF != ord('q'):
        choosen_ind = [np.random.choice(imgIDs)]
        sample_img_id = coco.getImgIds(imgIds=choosen_ind)
        sample_ann_id = coco.getAnnIds(imgIds=choosen_ind)

        img_infos = coco.loadImgs(sample_img_id)
        ann_infos = coco.loadAnns(sample_ann_id)

        img_path = os.path.join(TRAIN_IMAGE_PATH, img_infos[0]['file_name'])

        img = cv2.imread(img_path)
        color = (255, 255, 255)

        bboxes = []
        cat_names = []
        for ann in ann_infos:
            bboxes.append(ann['bbox'])
            # print(bboxes)
            cat_names.append(coco.loadCats(ann['category_id'])[0]['name'])
            # print(cat_names)
            img = cv2.rectangle(img, (int(bboxes[-1][0]), int(bboxes[-1][1])),
                                (int(bboxes[-1][0] + bboxes[-1][2]), int(bboxes[-1][1] + bboxes[-1][2])),
                                color, 2)
            img = cv2.putText(img, cat_names[-1], (int(bboxes[-1][0]), int(bboxes[-1][1])), cv2.FONT_HERSHEY_COMPLEX,
                              1, color, 1)

        cv2.imshow('coco', img)
    cv2.destroyAllWindows()





