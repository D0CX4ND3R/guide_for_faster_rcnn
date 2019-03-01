import tensorflow as tf
from tensorflow.contrib import slim


def faster_rcnn(features, rois, image_shape):
    # ROI Pooling
    roi_features = roi_pooling(features, rois, image_shape)


def process_faster_rcnn():
    pass


def roi_pooling(features, rois, image_shape):
    img_h, img_w = tf.cast(image_shape[0], tf.float32), tf.cast(image_shape[1], tf.float32)
    N = tf.shape(rois)[0]

    normalized_rois = _normalize_rois(rois, img_h, img_w)

    # TODO: Add settings for crop size ROI_SIZE, roi pooling kernel size ROI_POOLING_KERNEL_SIZE
    roi_size = 14
    roi_pooling_kernel_size = 2
    cropped_roi_features = tf.image.crop_and_resize(features, normalized_rois, tf.zeros((N,), tf.int32),
                                                    crop_size=[roi_size, roi_size])

    roi_features = slim.max_pool2d(cropped_roi_features, [roi_pooling_kernel_size, roi_pooling_kernel_size],
                                   stride=roi_pooling_kernel_size)
    return roi_features



def _normalize_rois(rois, img_h, img_w):
    x1, y1, x2, y2 = tf.unstack(rois, axis=1)

    normalized_x1 = x1 / img_w
    normalized_y1 = y1 / img_h
    normalized_x2 = x2 / img_w
    normalized_y2 = y2 / img_h

    normalized_rois = tf.transpose(tf.stack([normalized_x1, normalized_y1, normalized_x2, normalized_y2]))

    return tf.stop_gradient(normalized_rois)
