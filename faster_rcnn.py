import sys
from importlib import import_module

import tensorflow as tf
from tensorflow.contrib import slim

from utils.anchor_utils import decode_bboxes
from utils.losses import smooth_l1_loss_rcnn

import faster_rcnn_configs as frc


def faster_rcnn(features, rois, image_shape, is_training=True):
    with tf.variable_scope('rcnn'):
        # ROI Pooling
        roi_features = roi_pooling(features, rois, image_shape)

        if 'backbones' not in sys.path:
            sys.path.append('backbones')
        cnn = import_module(frc.BACKBONE, package='backbones')
        # Fully connected
        net_flatten = cnn.head(roi_features)

        with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(0.0005),
                            weights_initializer=slim.variance_scaling_initializer(1.0, mode='FAN_AVG', uniform=True),
                            activation_fn=None, trainable=is_training):
            cls_score = slim.fully_connected(net_flatten, frc.NUM_CLS + 1, scope='cls_fc')
            bbox_pred = slim.fully_connected(net_flatten, 4 * (frc.NUM_CLS + 1), scope='reg_fc')

            cls_score = tf.reshape(cls_score, [-1, frc.NUM_CLS + 1])
            bbox_pred = tf.reshape(bbox_pred, [-1, 4 * (frc.NUM_CLS + 1)])

    return cls_score, bbox_pred


def process_faster_rcnn(rois, bbox_pred, scores, image_shape):
    with tf.variable_scope('postprocess_faster_rcnn'):
        rois = tf.stop_gradient(rois)
        bbox_pred = tf.reshape(bbox_pred, [-1, frc.NUM_CLS + 1, 4])
        bbox_pred = tf.stop_gradient(bbox_pred)
        scores = tf.stop_gradient(scores)

        bboxes_pred_list = tf.unstack(bbox_pred, axis=1)
        score_list = tf.unstack(scores, axis=1)

        all_cls_bboxex = []
        all_cls_scores = []
        categories = []

        for i in range(frc.NUM_CLS + 1):
            encoded_bbox = bboxes_pred_list[i]
            score = score_list[i]

            decoded_bbox = decode_bboxes(encoded_bbox, rois, scale_factor=None)     # frc.ROI_SCALE_FACTORS

            # clip bounding to image shape
            predict_x_min, predict_y_min, predict_x_max, predict_y_max = tf.unstack(decoded_bbox, axis=1)
            image_height, image_width = tf.to_float(image_shape[0]), tf.to_float(image_shape[1])
            predict_x_min = tf.maximum(0., tf.minimum(image_width - 1, predict_x_min))
            predict_y_min = tf.maximum(0., tf.minimum(image_height - 1, predict_y_min))

            predict_x_max = tf.maximum(0., tf.minimum(image_width - 1, predict_x_max))
            predict_y_max = tf.maximum(0., tf.minimum(image_height - 1, predict_y_max))

            predict_bboxes = tf.transpose(tf.stack([predict_x_min, predict_y_min, predict_x_max, predict_y_max]))

            # NMS
            keep_ind = tf.image.non_max_suppression(predict_bboxes, score,
                                                    frc.FASTER_RCNN_NMS_MAX_BOX_PER_CLASS,
                                                    frc.FASTER_RCNN_NMS_IOU_THRESHOLD)

            per_cls_boxes = tf.gather(predict_bboxes, keep_ind)
            per_cls_scores = tf.gather(score, keep_ind)

            all_cls_bboxex.append(per_cls_boxes)
            all_cls_scores.append(per_cls_scores)
            categories.append(tf.ones_like(per_cls_scores) * i)

        final_bboxes = tf.concat(all_cls_bboxex, axis=0)
        final_scores = tf.concat(all_cls_scores, axis=0)
        final_categories = tf.concat(categories, axis=0)

        return final_bboxes, final_scores, final_categories


def roi_pooling(features, rois, image_shape):
    img_h, img_w = tf.cast(image_shape[0], tf.float32), tf.cast(image_shape[1], tf.float32)
    N = tf.shape(rois)[0]

    normalized_rois = _normalize_rois(rois, img_h, img_w)

    cropped_roi_features = tf.image.crop_and_resize(features, normalized_rois, tf.zeros((N,), tf.int32),
                                                    crop_size=[frc.FASTER_RCNN_ROI_SIZE, frc.FASTER_RCNN_ROI_SIZE])

    roi_features = slim.max_pool2d(cropped_roi_features,
                                   kernel_size=[frc.FASTER_RCNN_POOL_KERNEL_SIZE, frc.FASTER_RCNN_POOL_KERNEL_SIZE],
                                   stride=frc.FASTER_RCNN_POOL_KERNEL_SIZE)
    return roi_features


def _normalize_rois(rois, img_h, img_w):
    x1, y1, x2, y2 = tf.unstack(rois, axis=1)

    normalized_x1 = x1 / img_w
    normalized_y1 = y1 / img_h
    normalized_x2 = x2 / img_w
    normalized_y2 = y2 / img_h

    normalized_rois = tf.transpose(tf.stack([normalized_x1, normalized_y1, normalized_x2, normalized_y2]))

    return tf.stop_gradient(normalized_rois)


def build_faster_rcnn_losses(bbox_pred, bbox_targets, cls_score, labels, num_cls):
    bbox_loss = smooth_l1_loss_rcnn(bbox_pred, bbox_targets, labels, num_cls)
    cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=labels))
    return bbox_loss, cls_loss
