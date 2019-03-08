import os
import sys
import time
from importlib import import_module

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from toy_dataset.shape_generator import generate_shape_image
from region_proposal_network import rpn
from faster_rcnn import faster_rcnn, process_faster_rcnn, build_faster_rcnn_losses

from utils.image_draw import draw_rectangle_with_name, draw_rectangle
import faster_rcnn_configs as frc


def _network(inputs, image_shape, gt_bboxes):
    if 'backbones' not in sys.path:
        sys.path.append('backbones')
    cnn = import_module(frc.BACKBONE, package='backbones')
    # CNN
    feature_map = cnn.inference(inputs)

    features = slim.conv2d(feature_map, 512, [3, 3], normalizer_fn=slim.batch_norm,
                           normalizer_params={'decay': 0.995, 'epsilon': 0.0001},
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           scope='rpn_feature')

    # RPN
    rpn_cls_loss, rpn_cls_acc, rpn_bbox_loss, rois, labels, bbox_targets = rpn(features, image_shape, gt_bboxes)

    # TODO: Add summary
    class_names = ['BG', 'circle', 'rectangle', 'triangle']
    display_rois_img = tf.reshape(inputs, shape=[frc.IMAGE_SHAPE[0], frc.IMAGE_SHAPE[1], 3])
    for i in range(frc.NUM_CLS + 1):
        display_indices = tf.reshape(tf.where(tf.equal(labels, i)), [-1])
        display_rois = tf.gather(rois, display_indices)
        display_img = tf.py_func(draw_rectangle, [display_rois_img, display_rois], [tf.uint8])
        tf.summary.image('class_rois/{}'.format(class_names[i]), display_img)

    # RCNN
    cls_score, bbox_pred = faster_rcnn(features, rois, image_shape)
    cls_prob = slim.softmax(cls_score)
    cls_categories = tf.cast(tf.argmax(cls_prob, axis=1), dtype=tf.int32)
    rcnn_cls_acc = tf.reduce_mean(tf.cast(tf.equal(cls_categories, tf.cast(labels, tf.int32)), tf.float32))

    final_bbox, final_score, final_categories = process_faster_rcnn(rois, bbox_pred, cls_prob, image_shape)

    rcnn_bbox_loss, rcnn_cls_loss = build_faster_rcnn_losses(bbox_pred, bbox_targets, cls_prob, labels, frc.NUM_CLS + 1)

    loss_dict = {'rpn_cls_loss': rpn_cls_loss,
                 'rpn_bbox_loss': rpn_bbox_loss,
                 'rcnn_cls_loss': rcnn_cls_loss,
                 'rcnn_bbox_loss': rcnn_bbox_loss}
    acc_dict = {'rpn_cls_acc': rpn_cls_acc,
                'rcnn_cls_acc': rcnn_cls_acc}

    return final_bbox, final_score, final_categories, loss_dict, acc_dict


def _image_batch(image_shape=None, batch_size=1):
    if image_shape is None:
        image_shape = [224, 224]

    batch_image, bboxes, labels, _ = generate_shape_image(image_shape)

    batch_image = batch_image.reshape((batch_size, image_shape[0], image_shape[1], 3))

    return batch_image, np.hstack([bboxes, labels[:, np.newaxis]])


def _preprocess(inputs, image_shape=None):
    return inputs


def _main():
    tf_images = tf.placeholder(dtype=tf.float32,
                               shape=[frc.IMAGE_BATCH_SIZE, frc.IMAGE_SHAPE[0], frc.IMAGE_SHAPE[1], 3],
                               name='images')

    tf_labels = tf.placeholder(dtype=tf.int32, shape=[None, 5], name='ground_truth_bbox')

    tf_shape = tf.placeholder(dtype=tf.int32, shape=[None], name='image_shape')

    preprocessed_inputs = _preprocess(tf_images)

    final_bbox, final_score, final_categories, loss_dict, acc_dict = _network(preprocessed_inputs, tf_shape, tf_labels)

    display_indices_25 = tf.reshape(tf.where(tf.greater_equal(final_score, 0.25) &
                                             tf.less(final_score, 0.5) &
                                             tf.not_equal(final_categories, 0)), [-1])
    display_indices_50 = tf.reshape(tf.where(tf.greater_equal(final_score, 0.5) &
                                             tf.less(final_score, 0.75) &
                                             tf.not_equal(final_categories, 0)), [-1])
    display_indices_75 = tf.reshape(tf.where(tf.greater_equal(final_score, 0.75) &
                                             tf.not_equal(final_categories, 0)), [-1])
    display_bboxes_25 = tf.gather(final_bbox, display_indices_25)
    display_bboxes_50 = tf.gather(final_bbox, display_indices_50)
    display_bboxes_75 = tf.gather(final_bbox, display_indices_75)
    display_categories_25 = tf.gather(final_categories, display_indices_25)
    display_categories_50 = tf.gather(final_categories, display_indices_50)
    display_categories_75 = tf.gather(final_categories, display_indices_75)

    cls_names = [u'BG', u'circle', u'rectangle', u'triangle']
    display_image_25 = tf.py_func(draw_rectangle_with_name,
                                  [tf_images[0], display_bboxes_25, display_categories_25, cls_names],
                                  [tf.uint8])
    display_image_50 = tf.py_func(draw_rectangle_with_name,
                                  [tf_images[0], display_bboxes_50, display_categories_50, cls_names],
                                  [tf.uint8])
    display_image_75 = tf.py_func(draw_rectangle_with_name,
                                  [tf_images[0], display_bboxes_75, display_categories_75, cls_names],
                                  [tf.uint8])
    display_image_gt = tf.py_func(draw_rectangle_with_name,
                                  [tf_images[0], tf_labels[:, :-1], tf_labels[:, -1], cls_names],
                                  [tf.uint8])

    total_loss = frc.RPN_CLASSIFICATION_LOSS_WEIGHTS * loss_dict['rpn_cls_loss'] + \
                 frc.RPN_LOCATION_LOSS_WEIGHTS * loss_dict['rpn_bbox_loss'] + \
                 frc.FASTER_RCNN_CLASSIFICATION_LOSS_WEIGHTS * loss_dict['rcnn_cls_loss'] + \
                 frc.FASTER_RCNN_LOCATION_LOSS_WEIGHTS * loss_dict['rcnn_bbox_loss'] + \
                 tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    global_step = slim.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(learning_rate=0.05, global_step=0, decay_steps=10, decay_rate=0.5)
    # train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(total_loss)
    train_op = optimizer.apply_gradients(gradients, global_step=global_step)

    with tf.name_scope('loss'):
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('rpn_cls_loss', loss_dict['rpn_cls_loss'])
        tf.summary.scalar('rpn_bbox_loss', loss_dict['rpn_bbox_loss'])
        tf.summary.scalar('rcnn_cls_loss', loss_dict['rcnn_cls_loss'])
        tf.summary.scalar('rcnn_bbox_loss', loss_dict['rcnn_bbox_loss'])
    with tf.name_scope('accuracy'):
        tf.summary.scalar('rpn_acc',  acc_dict['rpn_cls_acc'])
        tf.summary.scalar('rcnn_acc', acc_dict['rcnn_cls_acc'])

    tf.summary.image('detection/gt', display_image_gt)
    tf.summary.image('detection/25', display_image_25)
    tf.summary.image('detection/50', display_image_50)
    tf.summary.image('detection/75', display_image_75)

    summary_op = tf.summary.merge_all()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        start_time = time.strftime('%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(frc.SUMMARY_PATH, start_time)
        save_model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(save_model_dir):
            os.mkdir(log_dir)
            os.mkdir(save_model_dir)
        summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)

        for step in range(frc.MAXIMUM_ITERS):
            images, gt_bboxes = _image_batch(frc.IMAGE_SHAPE)
            feed_dict = {tf_images: images, tf_labels: gt_bboxes, tf_shape: frc.IMAGE_SHAPE}

            if step % frc.REFRESH_LOGS_ITERS != 0:
                _, global_step_ = sess.run([train_op, global_step], feed_dict)
            else:
                step_time = time.time()

                _, total_loss_, rpn_cls_loss_, rpn_bbox_loss_, rcnn_cls_loss_, rcnn_bbox_loss_, \
                rpn_cls_acc_, rcnn_cls_acc_, summary_str, global_step_ = \
                    sess.run([train_op, total_loss, loss_dict['rpn_cls_loss'], loss_dict['rpn_bbox_loss'],
                              loss_dict['rcnn_cls_loss'], loss_dict['rcnn_bbox_loss'],
                              acc_dict['rpn_cls_acc'], acc_dict['rcnn_cls_acc'], summary_op, global_step], feed_dict)

                step_time = time.time() - step_time

                print(f'Iter: {step}',
                      f'| total_loss: {total_loss_:.3}',
                      f'| rpn_cls_loss: {rpn_cls_loss_:.3}',
                      f'| rpn_bbox_loss: {rpn_bbox_loss_:.3}',
                      f'| rcnn_cls_loss: {rcnn_cls_loss_:.3}',
                      f'| rcnn_bbox_loss: {rcnn_bbox_loss_:.3}',
                      f'| rpn_cls_acc: {rpn_cls_acc_:.3}',
                      f'| rcnn_cls_acc: {rcnn_cls_acc_:.3}',
                      f'| time: {step_time:.3}s')

                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                saver.save(sess, os.path.join(save_model_dir, frc.MODEL_NAME), step)
    summary_writer.close()


if __name__ == '__main__':
    _main()
