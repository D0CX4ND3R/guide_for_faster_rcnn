import sys
from importlib import import_module

import tensorflow as tf
from tensorflow.contrib import slim

from utils.anchor_utils import decode_bboxes
from utils.losses import smooth_l1_loss_rcnn, smooth_l1_loss_rcnn_ohem

import faster_rcnn_configs as frc


def faster_rcnn(features, rois, is_training=True):
    """

    :param features: Features from CNN with shape of [BATCH_SIZE, FEATURE_MAP_HEIGHT, FEATURE_MAP_WIDTH, CHANNELS]
    :param rois: ROIS from RPN with shape of [BATCH_SIZE, FASTER_RCNN_MINIBATCH_SIZE // BATCH_SIZE, 4]
    :param is_training:
    :return:
    """
    with tf.variable_scope('rcnn'):
        # ROI Pooling
        # Get roi_features with shape of [FASTER_RCNN_MINIBATCH_SIZE, ...]
        roi_features = roi_pooling(features, rois)

        if 'backbones' not in sys.path:
            sys.path.append('backbones')
        cnn = import_module(frc.BACKBONE, package='backbones')
        # Fully connected
        # Get shape is [FASTER_RCNN_MINIBATCH_SIZE, feature_dim]
        net_fc = cnn.head(roi_features, is_training=True)
        # net_fc = slim.fully_connected(net_flatten, frc.NUM_CLS, activation_fn=None,
        #                               normalizer_fn=slim.batch_norm,
        #                               normalizer_params={'decay': 0.995, 'epsilon': 0.0001},
        #                               weights_regularizer=slim.l2_regularizer(frc.L2_WEIGHT), scope='fc')

        with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(frc.L2_WEIGHT),
                            weights_initializer=slim.variance_scaling_initializer(1.0, mode='FAN_AVG', uniform=True),
                            activation_fn=None, trainable=is_training):
            # Let class score in shape of [FASTER_RCNN_MINIBATCH_SIZE, NUM_CLS + 1]
            cls_score = slim.fully_connected(net_fc, frc.NUM_CLS + 1, scope='cls_fc')
            # Let bbox prediction in shape of [FASTER_RCNN_MINIBATCH_SIZE, 4 * (NUM_CLS + 1)]
            bbox_pred = slim.fully_connected(net_fc, 4 * (frc.NUM_CLS + 1), scope='reg_fc')

            cls_score = tf.reshape(cls_score, [-1, frc.NUM_CLS + 1])
            bbox_pred = tf.reshape(bbox_pred, [-1, 4 * (frc.NUM_CLS + 1)])

    return cls_score, bbox_pred


def batchwise_process_faster_rcnn(rois, bbox_pred, scores, image_shape):
    def _instance_process(instance_rois, instance_bbox_pred, instance_scores, instance_image_shape):
        bboxes_pred_list = tf.unstack(instance_bbox_pred, axis=1)
        score_list = tf.unstack(instance_scores, axis=1)

        all_cls_bboxex = []
        all_cls_scores = []
        categories = []

        for i in range(frc.NUM_CLS + 1):
            encoded_bbox = bboxes_pred_list[i]
            score = score_list[i]

            decoded_bbox = decode_bboxes(encoded_bbox, instance_rois, scale_factor=None)     # frc.ROI_SCALE_FACTORS

            # clip bounding to image shape
            predict_x_min, predict_y_min, predict_x_max, predict_y_max = tf.unstack(decoded_bbox, axis=1)
            image_height, image_width = tf.to_float(instance_image_shape[0]), tf.to_float(instance_image_shape[1])

            # Clip predict coordinates in image shape (exclude padding zeros).
            predict_x_min = tf.maximum(0., tf.minimum(image_width - 1, predict_x_min))
            predict_y_min = tf.maximum(0., tf.minimum(image_height - 1, predict_y_min))

            predict_x_max = tf.maximum(0., tf.minimum(image_width - 1, predict_x_max))
            predict_y_max = tf.maximum(0., tf.minimum(image_height - 1, predict_y_max))

            predict_bboxes = tf.stack([predict_x_min, predict_y_min, predict_x_max, predict_y_max], axis=1)

            # NMS
            keep_ind = tf.image.non_max_suppression(predict_bboxes, score,
                                                    frc.FASTER_RCNN_NMS_MAX_BOX_PER_CLASS,
                                                    frc.FASTER_RCNN_NMS_IOU_THRESHOLD)

            per_cls_bboxes = tf.gather(predict_bboxes, keep_ind)
            per_cls_scores = tf.gather(score, keep_ind)

            all_cls_bboxex.append(per_cls_bboxes)
            all_cls_scores.append(per_cls_scores)
            categories.append(i * tf.ones_like(per_cls_scores, dtype=tf.int32))

        final_bboxes = tf.reshape(tf.concat(all_cls_bboxex, axis=0), [-1, 4])
        final_scores = tf.reshape(tf.concat(all_cls_scores, axis=0), [-1])
        categories = tf.reshape(tf.concat(categories, axis=0), [-1])

        # assert_op = tf.assert_greater_equal(frc.FASTER_RCNN_OUTPUT_NUM_PER_IMAGE_IN_BATCH, tf.shape(final_scores)[0])
        # with tf.control_dependencies([assert_op]):
        # If obtained targets less than configure value, padding them. Otherwise random choice(Not available now).
        final_bboxes, final_scores, categories = tf.cond(
            tf.greater_equal(frc.FASTER_RCNN_OUTPUT_NUM_PER_IMAGE_IN_BATCH, tf.shape(categories)[0]),
            true_fn=lambda: _padding_output(final_bboxes, final_scores, categories),
            false_fn=lambda: (final_bboxes[:frc.FASTER_RCNN_OUTPUT_NUM_PER_IMAGE_IN_BATCH],
                              final_scores[:frc.FASTER_RCNN_OUTPUT_NUM_PER_IMAGE_IN_BATCH],
                              categories[:frc.FASTER_RCNN_OUTPUT_NUM_PER_IMAGE_IN_BATCH]))

        return final_bboxes, final_scores, categories

    def _padding_output(final_bboxes, final_scores, categories):
        """
        Padding output shape to guarantee the mapping function output shape in the same.
        Output categories: -1 means not care in later processing.
        :param final_bboxes:
        :param final_scores:
        :param categories:
        :return:
        """
        padding_base = frc.FASTER_RCNN_OUTPUT_NUM_PER_IMAGE_IN_BATCH - tf.shape(categories)[0]
        # bbox_padding = tf.constant([[0, padding_base], [0, 0]], dtype=tf.int32)
        # score_and_cate_padding = tf.constant([[0, padding_base]], dtype=tf.int32)
        score_and_cate_padding, _ = tf.required_space_to_batch_paddings((tf.shape(categories)[0],),
                                                                        (frc.FASTER_RCNN_OUTPUT_NUM_PER_IMAGE_IN_BATCH,))
        bbox_padding, _ = tf.required_space_to_batch_paddings((tf.shape(categories)[0], 4),
                                                              (frc.FASTER_RCNN_OUTPUT_NUM_PER_IMAGE_IN_BATCH, 4))
        padding_final_bboxes = tf.pad(final_bboxes, bbox_padding, constant_values=-1)
        padding_final_scores = tf.pad(final_scores, score_and_cate_padding, constant_values=-1)
        padding_categories = tf.pad(categories, score_and_cate_padding, constant_values=-1)
        return padding_final_bboxes, padding_final_scores, padding_categories

    with tf.variable_scope('postprocess_faster_rcnn'):
        rois = tf.reshape(rois, [frc.IMAGE_BATCH_SIZE, -1, 4])
        rois = tf.stop_gradient(rois)

        bbox_pred = tf.reshape(bbox_pred, [frc.IMAGE_BATCH_SIZE, frc.FASTER_RCNN_MINIBATCH_SIZE // frc.IMAGE_BATCH_SIZE, frc.NUM_CLS + 1, 4])
        bbox_pred = tf.stop_gradient(bbox_pred)

        scores = tf.reshape(scores, [frc.IMAGE_BATCH_SIZE, frc.FASTER_RCNN_MINIBATCH_SIZE // frc.IMAGE_BATCH_SIZE, frc.NUM_CLS + 1])
        scores = tf.stop_gradient(scores)

        # Get batch final bboxes and scores
        # in shape of [IMAGE_BATCH_SIZE, NUM_CLS + 1, FASTER_RCNN_NMS_MAX_BOX_PER_CLASS, 4 or 1]
        batch_final_bboxes, batch_final_scores, batch_categories = \
            tf.map_fn(lambda i: _instance_process(rois[i], bbox_pred[i], scores[i], image_shape[i]),
                      tf.range(frc.IMAGE_BATCH_SIZE, dtype=tf.int32), dtype=(tf.float32, tf.float32, tf.int32))

        batch_categories_list = tf.unstack(batch_categories, axis=0)
        # batch_final_bboxes_list = tf.unstack(batch_final_bboxes, axis=0)
        # batch_final_scores_list = tf.unstack(batch_final_scores, axis=0)
        batch_final_bboxes_list = []
        batch_final_scores_list = []

        for i, categories in enumerate(batch_categories_list):
            keep_ind = tf.where(tf.not_equal(categories, -1))
            # final_bboxes, final_scores, categories = tf.cond(tf.not_equal(tf.shape(keep_ind)[0], 0),
            #                                                  true_fn=lambda: (tf.gather(batch_final_bboxes[i, ...], keep_ind),
            #                                                                   tf.gather(batch_final_scores[i], keep_ind),
            #                                                                   tf.gather(categories, keep_ind)),
            #                                                  false_fn=(None, None, None))
            batch_final_bboxes_list.append(tf.reshape(tf.gather(batch_final_bboxes[i], keep_ind), [-1, 4]))
            batch_final_scores_list.append(tf.reshape(tf.gather(batch_final_scores[i], keep_ind), [-1]))
            batch_categories_list[i] = tf.reshape(tf.gather(categories, keep_ind), [-1])

        return batch_final_bboxes_list, batch_final_scores_list, batch_categories_list


# def process_faster_rcnn(rois, bbox_pred, scores, image_shape):
#     with tf.variable_scope('postprocess_faster_rcnn'):
#         rois = tf.stop_gradient(rois)
#         bbox_pred = tf.reshape(bbox_pred, [-1, frc.NUM_CLS + 1, 4])
#         bbox_pred = tf.stop_gradient(bbox_pred)
#         scores = tf.stop_gradient(scores)
#
#         bboxes_pred_list = tf.unstack(bbox_pred, axis=1)
#         score_list = tf.unstack(scores, axis=1)
#
#         all_cls_bboxex = []
#         all_cls_scores = []
#         categories = []
#
#         for i in range(frc.NUM_CLS + 1):
#             encoded_bbox = bboxes_pred_list[i]
#             score = score_list[i]
#
#             decoded_bbox = decode_bboxes(encoded_bbox, rois, scale_factor=None)     # frc.ROI_SCALE_FACTORS
#
#             # clip bounding to image shape
#             predict_x_min, predict_y_min, predict_x_max, predict_y_max = tf.unstack(decoded_bbox, axis=1)
#             image_height, image_width = tf.to_float(image_shape[0]), tf.to_float(image_shape[1])
#             predict_x_min = tf.maximum(0., tf.minimum(image_width - 1, predict_x_min))
#             predict_y_min = tf.maximum(0., tf.minimum(image_height - 1, predict_y_min))
#
#             predict_x_max = tf.maximum(0., tf.minimum(image_width - 1, predict_x_max))
#             predict_y_max = tf.maximum(0., tf.minimum(image_height - 1, predict_y_max))
#
#             predict_bboxes = tf.stack([predict_x_min, predict_y_min, predict_x_max, predict_y_max], axis=1)
#
#             # NMS
#             keep_ind = tf.image.non_max_suppression(predict_bboxes, score,
#                                                     frc.FASTER_RCNN_NMS_MAX_BOX_PER_CLASS,
#                                                     frc.FASTER_RCNN_NMS_IOU_THRESHOLD)
#
#             per_cls_boxes = tf.gather(predict_bboxes, keep_ind)
#             per_cls_scores = tf.gather(score, keep_ind)
#
#             all_cls_bboxex.append(per_cls_boxes)
#             all_cls_scores.append(per_cls_scores)
#             categories.append(tf.ones_like(per_cls_scores) * i)
#
#         final_bboxes = tf.concat(all_cls_bboxex, axis=0, name='final_bboxes')
#         final_scores = tf.concat(all_cls_scores, axis=0, name='final_scores')
#         final_categories = tf.concat(categories, axis=0, name='final_categories')
#
#     return final_bboxes, final_scores, final_categories


def build_faster_rcnn_losses(bbox_pred, bbox_targets, cls_score, labels, num_cls):
    with tf.variable_scope('rcnn_losses'):
        labels = tf.reshape(labels, [-1])
        bbox_targets = tf.reshape(bbox_targets, [-1, 4 * num_cls])

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=labels)

        if frc.FASTER_RCNN_MINIBATCH_SIZE == -1:
            bbox_loss, cls_loss = smooth_l1_loss_rcnn_ohem(bbox_pred, bbox_targets, cross_entropy, labels, num_cls,
                                                           batch_size=frc.OHEM_BATCH_SIZE)
        else:
            bbox_loss = smooth_l1_loss_rcnn(bbox_pred, bbox_targets, labels, num_cls)
            cls_loss = tf.reduce_mean(cross_entropy)
    return bbox_loss, cls_loss


def roi_pooling(features, rois):
    """

    :param features: Features from CNN with shape of [BATCH_SIZE, FEATURE_MAP_HEIGHT, FEATURE_MAP_WIDTH, CHANNELS]
    :param rois:
    :return:
    """
    with tf.variable_scope('roi_pooling'):
        img_h, img_w = tf.cast(frc.IMAGE_SHAPE[0], tf.float32), tf.cast(frc.IMAGE_SHAPE[1], tf.float32)

        rois_per_image = frc.FASTER_RCNN_MINIBATCH_SIZE // frc.IMAGE_BATCH_SIZE
        _, rois_batch_indices = tf.meshgrid(tf.range(rois_per_image, dtype=tf.int32),
                                            tf.range(frc.IMAGE_BATCH_SIZE, dtype=tf.int32))

        rois_batch_indices = tf.reshape(rois_batch_indices, [-1])
        # N = tf.shape(rois)[0]

        # Let the shape of rois in [FASTER_RCNN_MINIBATCH_SIZE, 4]
        rois = tf.reshape(rois, [-1, 4])
        normalized_rois = _normalize_rois(rois, img_h, img_w)

        # cropped_roi_features = tf.image.crop_and_resize(features, normalized_rois, tf.zeros((N,), tf.int32),
        #                                                 crop_size=[frc.FASTER_RCNN_ROI_SIZE, frc.FASTER_RCNN_ROI_SIZE])
        # Get cropped roi features
        # Have shape [FASTER_RCNN_MINIBATCH_SIZE, FASTER_RCNN_ROI_SIZE, FASTER_RCNN_ROI_SIZE, CHANNELS]
        cropped_roi_features = tf.image.crop_and_resize(features, normalized_rois, rois_batch_indices,
                                                        crop_size=[frc.FASTER_RCNN_ROI_SIZE, frc.FASTER_RCNN_ROI_SIZE])

        # Shape is [FASTER_RCNN_MINIBATCH_SIZE, ...]
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

    # normalized coordinates [y1, x1, y2, x2]
    normalized_rois = tf.stack([normalized_y1, normalized_x1, normalized_y2, normalized_x2], axis=1)

    return tf.stop_gradient(normalized_rois)
