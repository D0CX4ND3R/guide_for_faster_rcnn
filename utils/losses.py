import tensorflow as tf


def build_rpn_losses(rpn_cls_score, rpn_cls_prob, rpn_bbox_pred, rpn_bbox_targets, rpn_labels):
    """

    :param rpn_cls_score:
    :param rpn_cls_prob:
    :param rpn_bbox_pred:
    :param rpn_bbox_targets:
    :param rpn_labels:
    :return: rpn_cls_loss, rpn_cls_acc, rpn_bbox_loss
    """
    with tf.variable_scope('RPN_LOSSES'):
        # calculate class accuracy
        rpn_cls_category = tf.argmax(rpn_cls_prob, axis=1)  # get 0 or 1 to represent the background or foreground

        # exclude not care labels
        calculated_rpn_target_indexes = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1))[0], [-1])

        rpn_cls_category = tf.cast(tf.gather(rpn_cls_category, calculated_rpn_target_indexes), dtype=tf.float32)
        gt_labels = tf.cast(tf.gather(rpn_labels, calculated_rpn_target_indexes), dtype=tf.float32)

        # rpn class loss
        rpn_cls_acc = tf.reduce_mean(tf.equal(rpn_cls_category, gt_labels))
        rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_labels,
                                                                                     logits=rpn_cls_score,
                                                                                     name='rpn_cls_loss'))

        rpn_bbox_loss = _smooth_l1_loss_rpn(rpn_bbox_pred, rpn_bbox_targets, rpn_labels)

        return rpn_cls_loss, rpn_cls_acc, rpn_bbox_loss


def build_rcnn_loss(bbox_pred, bbox_targets, labels, num_cls, sigma=1.0):
    pass


def _smooth_l1_loss(bbox_pred, bbox_targets, sigma=1.0):
    """
    Refer jemmy li zengarden2009@gmail.com losses.py
    :param bbox_pred:
    :param bbox_targets:
    :param sigma:
    :return:
    """
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    abs_box_diff = tf.abs(box_diff)

    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_box_diff, 1. / sigma_2)))
    loss_box = tf.pow(box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign + \
               (abs_box_diff - (0.5 / sigma_2) * (1.0 - smoothL1_sign))
    return loss_box


def _smooth_l1_loss_rpn(bbox_pred, bbox_targets, rpn_labels, sigma=1.0):
    value = _smooth_l1_loss(bbox_pred, bbox_targets)
    value = tf.reduce_sum(value, axis=1)

    rpn_select = tf.where(tf.greater(rpn_labels, 0))
    selected_value = tf.gather(value, rpn_select)

    non_ignored_mask = tf.stop_gradient(1.0 - tf.to_float(tf.equal(rpn_labels, -1)))

    bbox_loss = tf.reduce_sum(selected_value) / tf.maximum(1.0, tf.reduce_sum(non_ignored_mask))
    return bbox_loss
