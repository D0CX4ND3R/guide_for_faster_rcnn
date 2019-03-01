import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import resnext50
from region_proposal_network import rpn


def _network(inputs, image_shape, gt_bboxes):
    with tf.name_scope('CNN'):
        feature_map = resnext50.inference(inputs)

        features = slim.conv2d(feature_map, 512, [3, 3], normalizer_fn=slim.batch_norm,
                               normalizer_params={'decacy': 0.995, 'epsilon': 0.0001},
                               weights_regularizer=tf.nn.l2_normalize(0.0005),
                               scope='rpn_feature')

        rpn_cls_loss, rpn_cls_acc, rpn_bbox_loss, rois, labels, bbox_targets = rpn(features, image_shape, gt_bboxes)


