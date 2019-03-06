import tensorflow as tf
from tensorflow.contrib import slim

import numpy as np

from utils.anchor_utils import encode_bboxes, generate_anchors
from utils.losses import smooth_l1_loss_rpn
from utils.image_draw import draw_rectangle

import faster_rcnn_configs as frc


def rpn(features, image_shape, gt_bboxes):
    with tf.variable_scope('rpn'):
        # rpn_cls_score
        rpn_cls_score = slim.conv2d(features, 2 * frc.ANCHOR_NUM, [1, 1],
                                    normalizer_fn=slim.batch_norm,
                                    normalizer_params={'decay': frc.RPN_BN_DECACY, 'epsilon': frc.RPN_BN_EPS},
                                    weights_regularizer=slim.l2_regularizer(frc.RPN_WEIGHTS_L2_PENALITY_FACTOR),
                                    activation_fn=None, scope='rpn_cls_score')
        rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
        rpn_cls_prob = slim.softmax(rpn_cls_score, scope='rpn_cls_pred')

        # rpn_bbox_pred
        rpn_bbox_pred = slim.conv2d(features, frc.ANCHOR_NUM * (frc.NUM_CLS + 1), [1, 1],
                                    activation_fn=None, scope='rpn_bbox_pred')
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 4])

        featuremap_height, featuremap_width = tf.shape(features)[1], tf.shape(features)[2]
        featuremap_height = tf.cast(featuremap_height, dtype=tf.float32)
        featuremap_width = tf.cast(featuremap_width, dtype=tf.float32)

        anchors = make_anchors_in_image(frc.ANCHOR_BASE_SIZE, featuremap_width, featuremap_height,
                                        feature_stride=frc.FEATURE_STRIDE)

        # # TODO: Add summary
        # display_anchors_img = tf.zeros(shape=[tf.to_int32(image_shape[0]), tf.to_int32(image_shape[1]), 3],
        #                                dtype=tf.float32)
        # display_anchors_img = tf.py_func(draw_rectangle, [display_anchors_img, anchors], [tf.uint8])
        # tf.summary.image('anchors', display_anchors_img)

        # generate labels and bboxes to train rpn
        rpn_bbox_targets, rpn_labels = tf.py_func(generate_rpn_labels, [anchors, gt_bboxes, image_shape],
                                                  [tf.float32, tf.float32])
        rpn_labels = tf.to_int32(rpn_labels)
        rpn_labels = tf.reshape(rpn_labels, [-1])
        rpn_bbox_targets = tf.reshape(rpn_bbox_targets, [-1, 4])

        # rpn_losses
        rpn_cls_loss, rpn_cls_acc, rpn_bbox_loss = build_rpn_losses(rpn_cls_score, rpn_cls_prob,
                                                                    rpn_bbox_pred, rpn_bbox_targets,
                                                                    rpn_labels)

        # Get RCNN rois
        with tf.control_dependencies([rpn_labels]):
            # process rpn proposals, including clip, decode, nms
            rois, roi_scores = process_rpn_proposals(anchors, rpn_cls_prob, rpn_bbox_pred, image_shape)
            rois, labels, bbox_targets = tf.py_func(process_proposal_targets, [rois, gt_bboxes],
                                                    [tf.float32, tf.int32, tf.float32])

            rois = tf.reshape(rois, [-1, 4])
            labels = tf.reshape(tf.to_int32(labels), [-1])
            bbox_targets = tf.reshape(bbox_targets, [-1, 4 * (frc.NUM_CLS + 1)])

        # TODO: Add summary
        display_rois_img = tf.zeros(shape=[tf.to_int32(image_shape[0]), tf.to_int32(image_shape[1]), 3])
        display_positive_indices = tf.reshape(tf.where(tf.equal(labels, 1)), [-1])
        display_negative_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])
        display_positive_rois = tf.gather(rois, display_positive_indices)
        display_negative_rois = tf.gather(rois, display_negative_indices)
        display_positive_img = tf.py_func(draw_rectangle, [display_rois_img, display_positive_rois],
                                          [tf.uint8])
        display_negetive_img = tf.py_func(draw_rectangle, [display_rois_img, display_negative_rois],
                                          [tf.uint8])
        tf.summary.image('rpn_positive_rois', display_positive_img)
        tf.summary.image('rpn_negative_rois', display_negetive_img)

    return rpn_cls_loss, rpn_cls_acc, rpn_bbox_loss, rois, labels, bbox_targets


def process_proposal_targets(rpn_rois, gt_bboxes):
    """
    Assign object detection proposals to ground truth. Produce proposal classification labels and
    bounding box regression targets.
    :param rpn_rois:
    :param gt_bboxes:
    :return:
    """
    # rpn rois: [x1, y1, x2, y2]
    # ground truth: [x1, y1, x2, y2, label]
    if frc.ADD_GT_BOX_TO_TRAIN:
        all_rois = np.concatenate([rpn_rois, gt_bboxes[:, :-1]], axis=0)
    else:
        all_rois = rpn_rois

    rois_per_image = np.inf if frc.FASTER_RCNN_MINIBATCH_SIZE == -1 else frc.FASTER_RCNN_MINIBATCH_SIZE

    fg_rois_per_image = np.round(frc.FASTER_RCNN_POSITIVE_RATE * rois_per_image)

    # Sample rois with classification labels and bounding box regression.
    # TODO: Add class count in config.
    # ALGORTHM:
    # 1. Calculates overlaps between rois and ground truth.
    # 2. Get the maximum overlap area for each roi and set it label equal to the ground truth.

    overlaps = get_overlaps_py(rpn_rois, gt_bboxes[:, :-1])
    max_overlaps_gt_indexes = np.argmax(overlaps, axis=1)
    max_overlaps = np.max(overlaps, axis=1)

    labels = gt_bboxes[max_overlaps_gt_indexes, -1]

    # 3. Set overlap iou larger than iou threshold as positive sample, and less than threshold as negative samples.
    # IOU > POSITIVE_THRESHOLD = 0.5 => POSITIVE
    # 0 = NEGATIVE_THRESHOLD < IOU < POSITIVE_THRESHOLD = 0.5 => NEGATIVE
    # 4. Let the total numbers of rois equal to the
    # ROIS_PER_IMAGE = FOREGROUND_ROIS_PER_IMAGE + BACKGROUND_ROIS_PER_IMAGE

    # chose foreground indices
    fg_indices = np.where(max_overlaps >= frc.FASTER_RCNN_IOU_POSITIVE_THRESHOLD)[0]
    fg_rois_per_image = np.minimum(fg_rois_per_image, fg_indices.size)
    if fg_indices.size > 0:
        fg_indices = np.random.choice(fg_indices, size=int(fg_rois_per_image), replace=False)

    # chose background indices
    bg_indices = np.where((max_overlaps < frc.FASTER_RCNN_IOU_POSITIVE_THRESHOLD) &
                          (max_overlaps >= frc.FASTER_RCNN_IOU_NEGATIVE_THRESHOLD))[0]
    bg_roi_per_image = rois_per_image - fg_rois_per_image
    bg_roi_per_image = np.minimum(bg_roi_per_image, bg_indices.size)

    if bg_indices.size > 0:
        bg_indices = np.random.choice(bg_indices, size=int(bg_roi_per_image), replace=False)

    keep_indices = np.append(fg_indices, bg_indices)
    labels = labels[keep_indices]

    # 5. Make the negative labels equal to 0.
    labels[int(fg_rois_per_image):] = 0
    rois = all_rois[keep_indices]

    # 6. Encodes bounding boxes to targets coordinates for bounding box regression.
    bbox_targets_data = encode_bboxes(rois, gt_bboxes[max_overlaps_gt_indexes[keep_indices], :-1])
    # bbox_targets_data = np.hstack([labels[:, np.newaxis], bbox_targets_data]).astype(np.float32, copy=False)

    bbox_targets = np.zeros((labels.size, 4 * (frc.NUM_CLS + 1)), dtype=np.float32)
    inds = np.where(labels > 0)[0]
    for i in inds:
        cls = labels[i]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[i, start:end] = bbox_targets_data[i, 0:]

    return rois, labels, bbox_targets


def process_rpn_proposals(anchors, rpn_cls_pred, rpn_bbox_pred, image_shape, scale_factor=None):
    # 1. Trans bboxes
    t_xcenters, t_ycenters, t_w, t_h = tf.unstack(rpn_bbox_pred, axis=1)

    if scale_factor:
        t_xcenters /= scale_factor[0]
        t_ycenters /= scale_factor[1]
        t_w /= scale_factor[2]
        t_h /= scale_factor[3]

    anchors_x_min, anchors_y_min, anchors_x_max, anchors_y_max = tf.unstack(anchors, axis=1)
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
    image_height, image_width = tf.to_float(image_shape[0]), tf.to_float(image_shape[1])
    predict_x_min = tf.maximum(0., tf.minimum(image_width-1, predict_x_min))
    predict_y_min = tf.maximum(0., tf.minimum(image_height-1, predict_y_min))

    predict_x_max = tf.maximum(0., tf.minimum(image_width-1, predict_x_max))
    predict_y_max = tf.maximum(0., tf.minimum(image_height-1, predict_y_max))

    predict_bboxes = tf.transpose(tf.stack([predict_x_min, predict_y_min, predict_x_max, predict_y_max]))

    predict_targets_count = tf.minimum(frc.RPN_TOP_K_NMS_TRAIN, tf.shape(predict_bboxes)[0])
    sorted_rpn_cls_pred, sorted_pred_indeces = tf.nn.top_k(rpn_cls_pred[:, 1], predict_targets_count)
    sorted_bounding_boxes = tf.gather(predict_bboxes, sorted_pred_indeces)

    # 3. NMS
    selected_bboxes_indeces = tf.image.non_max_suppression(sorted_bounding_boxes, sorted_rpn_cls_pred,
                                                           max_output_size=frc.RPN_PROPOSAL_MAX_TRAIN,
                                                           iou_threshold=frc.RPN_NMS_IOU_THRESHOLD)

    selected_bboxes = tf.gather(sorted_bounding_boxes, selected_bboxes_indeces)
    selected_scores = tf.gather(sorted_rpn_cls_pred, selected_bboxes_indeces)
    return selected_bboxes, selected_scores


def make_anchors_in_image(anchor_base, feature_width, feature_height, feature_stride):
    _anchors = generate_anchors(original_anchor=[1, 1, anchor_base - 1, anchor_base - 1])
    shift_x = tf.range(feature_width, dtype=tf.float32) * feature_stride
    shift_y = tf.range(feature_height, dtype=tf.float32) * feature_stride
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)

    shifts = tf.stack([tf.reshape(shift_x, [-1]), tf.reshape(shift_y, [-1]),
                       tf.reshape(shift_x, [-1]), tf.reshape(shift_y, [-1])],
                      axis=1)
    all_anchors = _anchors[tf.newaxis, :, :] + shifts[:, tf.newaxis, :]
    return tf.reshape(all_anchors, [-1, 4])


def check_anchors(all_anchors, image_shape, allowed_boader=0):
    image_height, image_width = image_shape
    inside_boader_indeces = np.where(
        (all_anchors[:, 0] >= -allowed_boader) &
        (all_anchors[:, 1] >= -allowed_boader) &
        (all_anchors[:, 2] < image_width + allowed_boader) &
        (all_anchors[:, 3] < image_height + allowed_boader)
    )[0]
    return inside_boader_indeces


def generate_rpn_labels(all_anchors, gt_bboxes, image_shape):
    def _unmap(data, count, indexes, fill=0):
        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=np.float32)
            ret.fill(fill)
            ret[indexes] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
            ret.fill(fill)
            ret[indexes, :] = data
        return ret

    inside_boarder_indices = check_anchors(all_anchors, image_shape)
    anchors = all_anchors[inside_boarder_indices, :]

    # labels: positive=1; negative=0; not_care=-1
    labels = np.empty((len(inside_boarder_indices),), dtype=np.float32)
    labels.fill(-1)

    overlaps = get_overlaps_py(anchors, gt_bboxes)
    # overlaps: cross ious for each anchors and gt_boxes
    # rows: Anchors Indexes
    # columns: Ground Truth Bounding Box Indexes
    # For example K anchors with N ground truth bbox
    # overlaps is matrix have shape of K * N

    # For each anchor, get gt_bbox indices have max overlap region.
    # Shape: K
    max_overlap_gt_indices = np.argmax(overlaps, axis=1)
    # Get the max overlap for each anchor
    max_overlaps_for_each_anchor = overlaps[np.arange(len(inside_boarder_indices), dtype=np.int32),
                                            max_overlap_gt_indices]

    # For each gt_bbox, get anchor indices have max overlap region.
    # Shape: N
    max_overlap_anchor_indices = np.argmax(overlaps, axis=0)
    # Get the max overlap for each gt_bbox
    max_overlaps_for_each_gt = overlaps[max_overlap_anchor_indices,
                                        np.arange(len(gt_bboxes), dtype=np.int32)]

    # Find anchors have max overlap region
    max_overlap_indices = np.where(overlaps == max_overlaps_for_each_gt)[0]

    # Set negative labels
    labels[max_overlaps_for_each_anchor < frc.RPN_IOU_NEGATIVE_THRESHOLD] = 0

    # Set positive labels
    labels[max_overlap_indices] = 1
    labels[max_overlaps_for_each_anchor >= frc.RPN_IOU_POSITIVE_THRESHOLD] = 1

    # Set foreground and background limitation
    labels = foreground_background_limit(labels)

    bbox_targets = encode_bboxes(anchors, gt_bboxes[max_overlap_gt_indices, :])

    labels = _unmap(labels, all_anchors.shape[0], inside_boarder_indices, fill=-1)
    bbox_targets = _unmap(bbox_targets, all_anchors.shape[0], inside_boarder_indices)

    bbox_targets = bbox_targets.reshape([-1, 4])
    labels = labels.reshape([-1, 1])

    return bbox_targets, labels


def foreground_background_limit(labels):
    fg_num = int(frc.RPN_MINIBATCH_SIZE * frc.RPN_FOREGROUND_FRACTION)
    bg_num = int(frc.RPN_MINIBATCH_SIZE - fg_num)

    fg_indexes = np.where(labels == 1)[0]
    bg_indexes = np.where(labels == 0)[0]

    if len(fg_indexes) > fg_num:
        disable_indexes = np.random.choice(fg_indexes, len(fg_indexes) - fg_num, replace=False)
        labels[disable_indexes] = -1

    if len(bg_indexes) > bg_num:
        disable_indexes = np.random.choice(bg_indexes, len(bg_indexes) - bg_num, replace=False)
        labels[disable_indexes] = -1

    return labels


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
        calculated_rpn_target_indexes = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), [-1])

        rpn_cls_category = tf.cast(tf.gather(rpn_cls_category, calculated_rpn_target_indexes), dtype=tf.int32)
        gt_labels = tf.cast(tf.gather(rpn_labels, calculated_rpn_target_indexes), dtype=tf.int32)

        # rpn class loss
        rpn_cls_acc = tf.reduce_mean(tf.cast(tf.equal(rpn_cls_category, gt_labels), dtype=tf.float32))

        rpn_cls_score = tf.gather(rpn_cls_score, calculated_rpn_target_indexes)
        rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_labels,
                                                                                     logits=rpn_cls_score,
                                                                                     name='rpn_cls_loss'))

        rpn_bbox_loss = smooth_l1_loss_rpn(rpn_bbox_pred, rpn_bbox_targets, rpn_labels)

    return rpn_cls_loss, rpn_cls_acc, rpn_bbox_loss


# if __name__ == '__main__':
#     image_shape = [224, 224]
#     anchors = generate_anchors()
#     gt_bbox =
#     generate_rpn_labels(anchors,)