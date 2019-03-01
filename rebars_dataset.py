import os
import pandas as pd
import cv2
import numpy as np


class Dataset(object):
    def __init__(self, dataset_path, label_path):
        self.dataset_path = dataset_path

        self.data_list = []
        df = pd.read_csv(label_path)
        image_list = df['ID'].unique()
        for image_name in image_list:
            bboxes = self._get_bboxes(df, str(image_name))
            self.data_list.append({'image_name': str(image_name),
                                   'bboxes': bboxes})

    @property
    def dataset_size(self):
        return len(self.data_list)

    def _get_bboxes(self, date_frame, image_name):
        bboxes = date_frame[' Detection'][date_frame['ID'] == image_name]
        return_bboxes = []
        for ind in bboxes.index:
            return_bboxes.append([int(d) for d in bboxes.loc[ind].split()])

        return return_bboxes

    def get_sample(self, ind):
        sample = self.data_list[ind]
        sample_path = os.path.join(self.dataset_path, sample['image_name'])
        return sample_path, sample['bboxes']

    def get_sample_info(self, ind):
        sample = self.data_list[ind]
        return sample['image_name'], sample['bboxes']


if __name__ == '__main__':
    train_dataset_path = '/media/wx/新加卷/datasets/datafountain/rebars/train_dataset'
    train_label_file = '/media/wx/新加卷/datasets/datafountain/rebars/train_labels.csv'

    dataset_train = Dataset(train_dataset_path, train_label_file)
    ind = np.random.choice(dataset_train.dataset_size)
    sample_path, bboxes = dataset_train.get_sample(ind)

    im = cv2.imread(sample_path)
    for bbox in bboxes:
        im = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), thickness=5)

    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', im)
    k = cv2.waitKey(0)

    if k == 27:
        cv2.destroyWindow('image')

