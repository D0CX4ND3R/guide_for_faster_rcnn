import os
import sys
from importlib import import_module

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from region_proposal_network import rpn
from faster_rcnn import faster_rcnn, process_faster_rcnn
from toy_dataset.shape_generator import generate_shape_image
from utils.image_draw import draw_rectangle_with_name

import faster_rcnn_configs as frc


def _main():
    with tf.name_scope('inputs'):
        tf_images = tf.placeholder(dtype=tf.float32,
                                   shape=[frc.IMAGE_BATCH_SIZE, frc.IMAGE_SHAPE[0], frc.IMAGE_SHAPE[1], 3],
                                   name='images')
        tf_labels = tf.placeholder(dtype=tf.int32, shape=[None, 5], name='ground_truth_bbox')
        tf_shape = tf.placeholder(dtype=tf.int32, shape=[None], name='image_shape')

    final_bboxes, final_scores, final_categories = _network(tf_images, tf_shape, tf_labels)

    selected_indices = tf.where(tf.greater_equal(final_scores, 0.9) & tf.not_equal(final_categories, 0))
    final_bboxes = tf.gather(final_bboxes, selected_indices)
    final_scores = tf.gather(final_scores, selected_indices)
    final_categories = tf.gather(final_categories, selected_indices)

    class_names = frc.CLS_NAMES + ['circle', 'rectangle', 'triangle']

    saver = tf.train.Saver()

    dirs = os.listdir(frc.SUMMARY_PATH)
    dirs.sort()
    checkpoint_path = tf.train.latest_checkpoint(os.path.join(frc.SUMMARY_PATH, dirs[-1], 'model'))

    with tf.Session() as sess:
        if checkpoint_path:
            print('Load model:', checkpoint_path)
            saver.restore(sess, checkpoint_path)
        else:
            raise ValueError('No available model.')

        while cv2.waitKey(2000) & 0xFF != ord('q'):
            images, gt_bboxes = _image_batch(frc.IMAGE_SHAPE)
            feed_dict = {tf_images: images, tf_labels: gt_bboxes, tf_shape: frc.IMAGE_SHAPE}

            bboxes, scores, categories = sess.run([final_bboxes, final_scores, final_categories],
                                                  feed_dict=feed_dict)

            images = np.uint8(images.reshape([frc.IMAGE_SHAPE[0], frc.IMAGE_SHAPE[1], 3]))
            images_pred = draw_rectangle_with_name(images, bboxes[scores > 0.95], categories[scores > 0.9], class_names)
            images_gt = draw_rectangle_with_name(images, gt_bboxes[:, :-1], gt_bboxes[:, -1], class_names)

            cv2.imshow('pred', images_pred)
            cv2.imshow('gt', images_gt)
        cv2.destroyAllWindows()


def _network(inputs, image_shape, gt_bboxes):
    if 'backbones' not in sys.path:
        sys.path.append('backbones')
    cnn = import_module(frc.BACKBONE, package='backbones')
    # CNN
    feature_map = cnn.inference(inputs)

    features = slim.conv2d(feature_map, 512, [3, 3], normalizer_fn=slim.batch_norm,
                           normalizer_params={'decay': 0.995, 'epsilon': 0.0001},
                           weights_regularizer=slim.l2_regularizer(frc.L2_WEIGHT),
                           scope='rpn_feature')

    # RPN
    _, _, _, rois, labels, bbox_targets = rpn(features, image_shape, gt_bboxes)

    # RCNN
    cls_score, bbox_pred = faster_rcnn(features, rois, image_shape)

    cls_prob = slim.softmax(cls_score)

    final_bbox, final_score, final_categories = process_faster_rcnn(rois, bbox_pred, cls_prob, image_shape)

    return final_bbox, final_score, final_categories


def _image_batch(image_shape=None, batch_size=1):
    if image_shape is None:
        image_shape = [224, 224]

    batch_image, bboxes, labels, _ = generate_shape_image(image_shape)

    batch_image = batch_image.reshape((batch_size, image_shape[0], image_shape[1], 3))

    return batch_image, np.hstack([bboxes, labels[:, np.newaxis]])


if __name__ == '__main__':
    _main()
