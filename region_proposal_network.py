import tensorflow as tf
from tensorflow.contrib import slim

import numpy as np


class RpnConfig(object):
    def __init__(self, class_count):
        self.scale = [0.5, 1, 2]
        self.size = [128, 256, 512]
        self.cls_count = class_count

    @property
    def anchor_num(self):
        return len(self.scale) * len(self.size)

    @property
    def class_count(self):
        return self.cls_count

    @property
    def rpn_iou_positive_threshold(self):
        return self.rpn_iou_positive_th


def _cls():
    pass


def _reg():
    pass


def generate_anchors(original_anchor=[0, 0, 15, 15], scales=[8, 16, 32], ratios=[0.5, 1, 2]):
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


def rpn(feature_map, rpn_config=RpnConfig(2)):
    with tf.variable_scope('rpn'):
        features = slim.conv2d(feature_map, 512, [3, 3], normalizer_fn=slim.batch_norm,
                               normalizer_params={'decacy': 0.995, 'epsilon': 0.0001},
                               weights_regularizer=tf.nn.l2_normalize(0.0005),
                               scope='rpn_feature')

        # rpn_cls_score
        rpn_cls_score = slim.conv2d(features, 2 * rpn_config.anchor_num, [1, 1],
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params={'decacy': 0.995, 'epsilon': 0.0001},
                                    weights_regularizer=tf.nn.l2_normalize(0.0005),
                                    activation_fn=None, scope='rpn_cls_score')
        rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
        rpn_cls_pred = slim.softmax(rpn_cls_score, scope='rpn_cls_pred')

        # rpn_bbox_pred
        rpn_bbox_pred = slim.conv2d(features, rpn_config.anchor_num * rpn_config.class_count, [1, 1],
                                    activation_fn=None, scope='rpn_bbox_pred')
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 4])

    return rpn_cls_pred, rpn_bbox_pred


def process_rpn_proposals(anchors, rpn_cls_pred, rpn_bbox_pred, image_shape, scale_factor=None):
    # 1. Trans bboxes
    t_xcenters, t_ycenters, t_w, t_h = tf.unstack(rpn_bbox_pred, axis=1)

    if scale_factor:
        t_xcenters /= scale_factor[0]
        t_ycenters /= scale_factor[1]
        t_w /= scale_factor[2]
        t_h /= scale_factor[3]

    anchors_x_min, anchors_y_mgenerate_anchorsin, anchors_x_max, anchors_y_max = tf.unstack(anchors, axis=1)
    anchors_width = anchors_x_max - anchors_x_min
    anchors_height = anchors_y_max - anchors_y_min
    anchors_center_x = anchors_x_min + anchors_width / 2.0
    anchors_center_y = anchors_y_min + anchors_height / 2.0

    # According to equations in Faster-RCNN
    # t_x = (x - x_a) / w_a
    # t_y = (y - y_a) / h_a
    # t_w = log(w / w_a)
    # t_h = log(h / h_a)
    # Where (t_x, t_y, t_w, t_h) is the prediction by RPN, (x, y, w, h) is the prediction of bounding box,
    # (x_a, y_a, w_a, h_a) is the generated anchor box.
    # The RPN will be optimized to calculate the coordinates of (t_x, t_y, t_w, t_h).
    predict_center_x = t_xcenters * anchors_width + anchors_center_x
    predict_center_y = t_ycenters * anchors_height + anchors_center_y
    predict_width = tf.exp(t_w) * anchors_width
    predict_height = tf.exp(t_h) * anchors_height

    predict_x_min = predict_center_x - predict_width / 2
    predict_y_min = predict_center_y - predict_height / 2
    predict_x_max = predict_center_x + predict_width / 2
    predict_y_max = predict_center_y + predict_height / 2

    # 2. Clip bounding boxes, make all boxes in the bounding of image
    image_height, image_width = image_shape
    predict_x_min = tf.maximum(0, tf.minimum(image_width-1, predict_x_min))
    predict_y_min = tf.maximum(0, tf.minimum(image_height-1, predict_y_min))

    predict_x_max = tf.maximum(0, tf.minimum(image_width-1, predict_x_max))
    predict_y_max = tf.maximum(0, tf.minimum(image_height-1, predict_y_max))

    predict_bboxes = tf.transpose(tf.stack([predict_x_min, predict_y_min, predict_x_max, predict_y_max]))

    predict_targets_count = tf.minimum(12000, tf.shape(predict_bboxes)[0])
    sorted_cls_scores, sorted_pred_indeces = tf.nn.top_k(rpn_cls_pred, predict_targets_count)
    sorted_bounding_boxes = tf.gather(predict_bboxes, sorted_pred_indeces)
    selected_bboxes_indeces = tf.image.non_max_suppression(sorted_bounding_boxes, sorted_cls_scores,
                                                           image_shape[0] * image_shape[1])

    selected_bboxes = tf.gather(sorted_bounding_boxes, selected_bboxes_indeces)
    selected_scores = tf.gather(sorted_cls_scores, selected_bboxes_indeces)
    return selected_bboxes, selected_scores


def make_anchors_in_image(anchor_base, feature_width, feature_height, feature_stride):
    _anchors = generate_anchors(original_anchor=[1, 1, anchor_base - 1, anchor_base - 1])
    shift_x = np.arange(feature_width) * feature_stride
    shift_y = np.arange(feature_height) * feature_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack([shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()]).transpose()
    all_anchors = _anchors[np.newaxis, :, :] + shifts[:, np.newaxis, :]
    return all_anchors.reshape(-1, 4)


def check_anchors(all_anchors, image_shape, allowed_boader=0):
    image_height, image_width = image_shape
    inside_boader_indeces = np.where(
        (all_anchors[:, 0] >= -allowed_boader) &
        (all_anchors[:, 1] >= -allowed_boader) &
        (all_anchors[:, 2] < image_width + allowed_boader) &
        (all_anchors[:, 3] < image_height + allowed_boader)
    )[0]
    return inside_boader_indeces


def generate_rpn_labels(all_anchors, gt_bboxes):
    inside_boarder_indexes = check_anchors(all_anchors)
    anchors = all_anchors[inside_boarder_indexes, :]

    # labels: positive=1; negative=0; not_care=-1
    labels = np.empty((len(inside_boarder_indexes),), dtype=np.float32)
    labels.fill(-1)

    overlaps = get_overlaps_py(anchors, gt_bboxes)
    # overlaps: cross ious for each anchors and gt_boxes
    # rows: Anchors Indexes
    # columns: Ground Truth Bounding Box Indexes
    # The indexes of ground truth bounding box have maximum overlap area for each anchors.
    max_overlap_gt_indexes = np.argmax(overlaps, axis=1)
    max_overlaps_gt = overlaps[np.arange(len(inside_boarder_indexes), dtype=np.int32), max_overlap_gt_indexes]

    # The indexes of anchors with maximum overlap area with each ground truth bonding boxes.
    max_overlap_anchor_indexes = np.argmax(overlaps, axis=0)
    max_overlap_anchor = overlaps[max_overlap_anchor_indexes, np.arange(len(gt_bboxes), dtype=np.int32)]
    max_overlap_indexes = np.where(max_overlap_anchor == max_overlaps_gt)[0]

    # TODO: Modify CONFIG
    # Set negative labels
    labels[max_overlaps_gt < 0.3] = 0

    # Set positive labels
    labels[max_overlaps_gt >= 0.7] = 1

    return anchors, labels


def foreground_background_limit(labels):
    fg_num = int(0.5 * 2000)
    bg_num = int(0.5 * 2000)
    # TODO: Set limitation number of foreground and background.
    fg_indexes = np.where(labels == 1)[0]
    bg_indexes = np.where(labels == 0)[0]

    if len(fg_indexes) > fg_num:
        disable_indexes = np.random.choice(fg_indexes, len(fg_indexes) - fg_num, replace=False)
        labels[disable_indexes] = -1

    if len(bg_indexes) > bg_num:
        disable_indexes = np.random.choice(bg_indexes, len(bg_indexes) - bg_num, replace=False)
        labels[disable_indexes] = -1


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


def process_anchors(rpn_cls_pred, gt_bboxes, img_shape, all_anchors):
    # Calculating the essential rpn labels, bboxes targets to train RPN by
    # rpn_class_score
    # ground_truth_boxes
    # image_dims
    # feat_stride (maybe means the zoom factor by backbone cnn)
    # anchor_scales
    # The anchor_target_layer is a well designed function to process this by many open source projects can be refer.
    im_h, im_w = img_shape



def get_verlaps(pred_bboxes, gt_bboxes):
    return tf.py_func(get_overlaps_py, [pred_bboxes, gt_bboxes], [tf.float32])


def get_overlaps_py(pred_bboxes, gt_bboxes):
    """
    Caluculate overlap area of predicted acnchors and ground truth. Inputs K anchors and N ground truth boxes, returns
    K * N array of ious.
    :param pred_bboxes: Generated anchors.
    :param gt_bboxes: Ground truth bounding boxes.
    :return: Intersection of Union of anchors and ground truth bounding box.
    """
    len_pred_bboxes = len(pred_bboxes)
    len_gt_bboxes = len(gt_bboxes)
    pred_map_indeces = np.arange(len_pred_bboxes, dtype=np.int32)
    pred_map_indeces = np.repeat(pred_map_indeces, (len_gt_bboxes,))
    gt_map_indeces = np.arange(len_gt_bboxes, dtype=np.int32)
    gt_map_indeces = np.repeat(gt_map_indeces[:, np.newaxis], (len_pred_bboxes,), axis=1).transpose().ravel()

    intersection_boxes_x1 = np.maximum(gt_bboxes[gt_map_indeces, 0], pred_bboxes[pred_map_indeces, 0])
    intersection_boxes_y1 = np.maximum(gt_bboxes[gt_map_indeces, 1], pred_bboxes[pred_map_indeces, 1])
    intersection_boxes_x2 = np.minimum(gt_bboxes[gt_map_indeces, 2], pred_bboxes[pred_map_indeces, 2])
    intersection_boxes_y2 = np.minimum(gt_bboxes[gt_map_indeces, 3], pred_bboxes[pred_map_indeces, 3])

    iws = intersection_boxes_x2 - intersection_boxes_x1 + 1
    ihs = intersection_boxes_y2 - intersection_boxes_y1 + 1

    less_zero_indeces = np.bitwise_or(iws < 0, ihs < 0)
    iws[less_zero_indeces] = 0
    ihs[less_zero_indeces] = 0
    gt_bboxes_areas = (gt_bboxes[gt_map_indeces, 2] - gt_bboxes[gt_map_indeces, 0] + 1) * \
                      (gt_bboxes[gt_map_indeces, 3] - gt_bboxes[gt_map_indeces, 1] + 1)
    pred_bboxes_areas = (pred_bboxes[pred_map_indeces, 2] - pred_bboxes[pred_map_indeces, 0] + 1) * \
                        (pred_bboxes[pred_map_indeces, 3] - pred_bboxes[pred_map_indeces, 1] + 1)
    intersection_areas = iws * ihs
    ious = intersection_areas / (gt_bboxes_areas + pred_bboxes_areas - intersection_areas)

    return ious.reshape([len_pred_bboxes, len_gt_bboxes])



def build_rpn_loss(rois, roi_scores, iou_threshold, top_k_nms, top_proposal):
    pass


def build_cls_loss():
    pass


def build_loss():
    pass


if __name__ == '__main__':
    a = generate_anchors([0, 0, 15, 15])
    import matplotlib.pyplot as plt


    def draw_rectangle(ax, anchor, color):
        x1 = anchor[0]
        y1 = anchor[1]
        x2 = anchor[2]
        y2 = anchor[3]
        ax.hlines(y1, x1, x2, colors=color)
        ax.hlines(y2, x1, x2, colors=color)
        ax.vlines(x1, y1, y2, colors=color)
        ax.vlines(x2, y1, y2, colors=color)


    colors = ['b', 'g', 'r'] * 3
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for anchor, color in zip(a, colors):
        draw_rectangle(ax, anchor, color)
        x_ctr = (anchor[0] + anchor[2]) / 2
        y_ctr = (anchor[1] + anchor[3]) / 2
        print('center:')
        print('x=', x_ctr, 'y=', y_ctr)
        print('area:', (anchor[2] - anchor[0] + 1) * (anchor[3] - anchor[1] + 1))
        print('w=', anchor[2] - anchor[0] + 1, 'h=', anchor[3] - anchor[1] + 1)
        ax.scatter(x_ctr, y_ctr, c=color)
    ax.set_aspect('equal')
    plt.show()

