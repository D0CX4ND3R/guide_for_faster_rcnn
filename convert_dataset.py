import os
from utils import Dataset


if __name__ == '__main__':
    dataset_dir = '/media/wx/新加卷/datasets'
    target_dir = os.path.join(dataset_dir, 'rebars_dataset')
    annotation_dir = os.path.join(target_dir, 'Annotations')
    name_dir = os.path.join(target_dir, 'Names')

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
        os.mkdir(annotation_dir)
        os.mkdir(name_dir)

    train_dataset_path = '/media/wx/新加卷/datasets/datafountain/rebars/train_dataset'
    train_label_file = '/media/wx/新加卷/datasets/datafountain/rebars/train_labels.csv'

    dataset_train = Dataset(train_dataset_path, train_label_file)
    total_sample_count = dataset_train.dataset_size

    train_file_path = os.path.join(name_dir, 'train.txt')
    train_file_writer = open(train_file_path, 'w')

    for i in range(total_sample_count):
        sample_name, bboxes = dataset_train.get_sample_info(i)
        annotation_file_path = os.path.join(annotation_dir, sample_name.split('.')[0] + '.txt')

        with open(annotation_file_path, 'w') as annotation_writer:
            for bbox in bboxes:
                bbox = [str(d) for d in bbox]
                annotation_writer.write(' '.join(bbox) + ' 1\n')
            train_file_writer.write(sample_name.split('.')[0] + '\n')

    train_file_writer.close()
