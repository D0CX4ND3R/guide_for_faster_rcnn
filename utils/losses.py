import tensorflow as tf


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


def smooth_l1_loss_rpn(bbox_pred, bbox_targets, rpn_labels, sigma=3.0):
    value = _smooth_l1_loss(bbox_pred, bbox_targets, sigma)
    value = tf.reduce_sum(value, axis=1)

    rpn_select = tf.where(tf.greater(rpn_labels, 0))
    selected_value = tf.gather(value, rpn_select)

    non_ignored_mask = tf.stop_gradient(1.0 - tf.to_float(tf.equal(rpn_labels, -1)))

    bbox_loss = tf.reduce_sum(selected_value) / tf.maximum(1.0, tf.reduce_sum(non_ignored_mask))
    return bbox_loss


def smooth_l1_loss_rcnn(bbox_pred, bbox_targets, label, num_classes, sigma=1.0):
    outside_mask = tf.stop_gradient(tf.to_float(tf.greater(label, 0)))

    bbox_pred = tf.reshape(bbox_pred, [-1, num_classes, 4])
    bbox_targets = tf.reshape(bbox_targets, [-1, num_classes, 4])

    value = _smooth_l1_loss(bbox_pred, bbox_targets, sigma)
    value = tf.reduce_sum(value, 2)
    value = tf.reshape(value, [-1, num_classes])

    inside_mask = tf.one_hot(tf.reshape(label, [-1, 1]),
                             depth=num_classes, axis=1)

    inside_mask = tf.stop_gradient(
        tf.to_float(tf.reshape(inside_mask, [-1, num_classes])))

    normalizer = tf.to_float(tf.shape(bbox_pred)[0])
    bbox_loss = tf.reduce_sum(
        tf.reduce_sum(value * inside_mask, 1) * outside_mask) / normalizer

    return bbox_loss
