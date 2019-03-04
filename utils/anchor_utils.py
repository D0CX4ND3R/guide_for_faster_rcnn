import numpy as np
import tensorflow as tf


def generate_anchors(original_anchor=None, scales=None, ratios=None):
    if ratios is None:
        ratios = [0.5, 1, 2]
    if scales is None:
        scales = [8, 16, 32]
    if original_anchor is None:
        original_anchor = [0, 0, 15, 15]

    def _gen_anchors(width, height):
        wn = np.repeat(width, len(scales)) * np.array(scales)
        hn = np.repeat(height, len(scales)) * np.array(scales)

        wn = wn[:, np.newaxis]
        hn = hn[:, np.newaxis]
        return np.hstack([x_center - (wn - 1) / 2, y_center - (hn - 1) / 2,
                          x_center + (wn - 1) / 2, y_center + (hn - 1) / 2])

    # calculate original anchor's width, height and center coordinate
    w = original_anchor[2] - original_anchor[0] + 1
    h = original_anchor[3] - original_anchor[1] + 1
    x_center = original_anchor[0] + (w - 1) / 2
    y_center = original_anchor[1] + (h - 1) / 2

    # calculate the original anchor's area
    original_area = w * h

    # calculate the three ratios areas
    three_ratios_area = original_area * np.array(ratios)

    # calculate the three kinds of areas width and height
    three_ratios_width = np.round(np.sqrt(three_ratios_area))
    three_ratios_height = three_ratios_width / np.array(ratios)

    # calculate anchors, each anchors coordinate is [x1, y1, x2, y2]
    # (x1, y1)-----------------------------------------
    # |                                                |
    # |                                                |
    # |                                                |
    # |                                                |
    # |                                                |
    # ------------------------------------------(x2, y2)
    anchors = [_gen_anchors(wr, hr) for wr, hr in zip(three_ratios_width, three_ratios_height)]

    return np.vstack(anchors)


def bboxes2anchors(bboxes):
    widths = bboxes[:, 2] - bboxes[:, 0] + 1
    heights = bboxes[:, 3] - bboxes[:, 1] + 1
    x_centers = (bboxes[:, 2] + bboxes[:, 0]) / 2.0
    y_centers = (bboxes[:, 3] + bboxes[:, 1]) / 2.0
    return x_centers, y_centers, widths, heights


def anchors2bboxes(anchors):
    xx1 = anchors[:, 0] - anchors[:, 2] / 2.0 + 0.5
    xx2 = anchors[:, 0] + anchors[:, 2] / 2.0 - 0.5
    yy1 = anchors[:, 1] - anchors[:, 3] / 2.0 + 0.5
    yy2 = anchors[:, 1] + anchors[:, 3] / 2.0 + 0.5
    return xx1, yy1, xx2, yy2


def encode_bboxes(pred_bboxes, gt_bboxes, scale_factor=None):
    pred_x_centers, pred_y_centers, pred_widths, pred_heigths = bboxes2anchors(pred_bboxes)
    gt_x_centers, gt_y_centers, gt_widths, gt_heigths = bboxes2anchors(gt_bboxes)

    # Avoid divide zero
    pred_widths = pred_widths + 1e-8
    pred_heigths = pred_heigths + 1e-8
    gt_widths = gt_widths + 1e-8
    gt_heigths = gt_heigths + 1e-8

    t_x = (pred_x_centers - gt_x_centers) / gt_x_centers
    t_y = (pred_y_centers - gt_y_centers) / gt_y_centers
    t_w = np.log(pred_widths / gt_widths)
    t_h = np.log(pred_heigths / gt_heigths)

    if scale_factor:
        t_x *= scale_factor[0]
        t_y *= scale_factor[1]
        t_w *= scale_factor[2]
        t_h *= scale_factor[3]
    return np.stack([t_x, t_y, t_w, t_h], axis=1)


def decode_bboxes(encoded_pred_bboxes, gt_bboxes, scale_factor=None):
    t_x, t_y, t_w, t_h = tf.unstack(encoded_pred_bboxes, axis=1)
    if scale_factor:
        t_x = t_x / scale_factor[0]
        t_y = t_y / scale_factor[1]
        t_w = t_w / scale_factor[2]
        t_h = t_h / scale_factor[3]

    gt_x_centers, gt_y_centers, gt_widths, gt_heigths = bboxes2anchors(gt_bboxes)

    pred_x_centers = t_x * gt_x_centers + gt_x_centers
    pred_y_centers = t_y * gt_y_centers + gt_y_centers
    pred_widths = tf.exp(t_w) * gt_widths
    pred_heights = tf.exp(t_h) * gt_heigths

    pred_xx1, pred_yy1, pred_xx2, pred_yy2 = anchors2bboxes(tf.stack(
        [pred_x_centers, pred_y_centers, pred_widths, pred_heights], axis=1))

    return tf.stack([pred_xx1, pred_yy1, pred_xx2, pred_yy2], axis=1)
