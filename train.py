import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from toy_dataset.shape_generator import generate_shape_image
import resnext50
from region_proposal_network import rpn
from faster_rcnn import faster_rcnn, process_faster_rcnn, build_faster_rcnn_losses

import faster_rcnn_configs as frc


def _network(inputs, image_shape, gt_bboxes):
    with tf.name_scope('CNN'):
        feature_map = resnext50.inference(inputs)

        features = slim.conv2d(feature_map, 512, [3, 3], normalizer_fn=slim.batch_norm,
                               normalizer_params={'decacy': 0.995, 'epsilon': 0.0001},
                               weights_regularizer=tf.nn.l2_normalize(0.0005),
                               scope='rpn_feature')

    with tf.name_scope('RPN'):
        rpn_cls_loss, rpn_cls_acc, rpn_bbox_loss, rois, labels, bbox_targets = rpn(features, image_shape, gt_bboxes)

    with tf.name_scope('RCNN'):
        cls_score, bbox_pred = faster_rcnn(features, rois, image_shape)
        cls_prob = slim.softmax(cls_score)
        cls_categories = tf.argmax(cls_prob, axis=1)
        rcnn_cls_acc = tf.reduce_mean(tf.cast(tf.equal(cls_categories, tf.cast(labels, tf.int32)), tf.float32))

        final_bbox, final_score, final_categories = process_faster_rcnn(rois, bbox_pred, cls_prob, image_shape)

        # TODO: Add num cls
        num_cls = 3
        rcnn_bbox_loss, rcnn_cls_loss = build_faster_rcnn_losses(bbox_pred, bbox_targets, cls_prob, labels, num_cls + 1)

        loss_dict = {'rpn_cls_loss': rpn_cls_loss,
                     'rpn_bbox_loss': rpn_bbox_loss,
                     'rcnn_cls_loss': rcnn_cls_loss,
                     'rcnn_bbox_loss': rcnn_bbox_loss}
        acc_dict = {'rpn_cls_acc': rpn_cls_acc,
                    'rcnn_cls_acc': rcnn_cls_acc}

    return final_bbox, final_score, final_categories, loss_dict, acc_dict


def _image_batch(image_shape=None, batch_size=1):
    if image_shape is None:
        image_shape = [448, 448]

    batch_image, bboxes, labels, _ = generate_shape_image(image_shape, 1)

    return batch_image, np.hstack([bboxes, labels[:, np.newaxis]])


def _main():
    image_shape = [448, 448]
    images, gt_bboxes = _image_batch(image_shape)

    final_bbox, final_score, final_categories, loss_dict, acc_dict = _network(images, image_shape, gt_bboxes)

