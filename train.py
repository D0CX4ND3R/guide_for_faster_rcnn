import os
import time
from typing import Any, Union

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.contrib import slim

from toy_dataset.shape_generator import generate_shape_image
import resnext50
from region_proposal_network import rpn
from faster_rcnn import faster_rcnn, process_faster_rcnn, build_faster_rcnn_losses

import faster_rcnn_configs as frc


def _network(inputs, image_shape, gt_bboxes):
    # CNN
    feature_map = resnext50.inference(inputs)

    features = slim.conv2d(feature_map, 512, [3, 3], normalizer_fn=slim.batch_norm,
                           normalizer_params={'decay': 0.995, 'epsilon': 0.0001},
                           weights_regularizer=slim.l2_regularizer(0.0005),
                           scope='rpn_feature')

    # RPN
    rpn_cls_loss, rpn_cls_acc, rpn_bbox_loss, rois, labels, bbox_targets = rpn(features, image_shape, gt_bboxes)

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


def _main():
    tf_images = tf.placeholder(dtype=tf.float32,
                               shape=[frc.IMAGE_BATCH_SIZE, frc.IMAGE_SHAPE[0], frc.IMAGE_SHAPE[1], 3],
                               name='images')

    tf_labels = tf.placeholder(dtype=tf.int32, shape=[None, 5], name='ground_truth_bbox')

    tf_shape = tf.placeholder(dtype=tf.int32, shape=[None], name='image_shape')

    final_bbox, final_score, final_categories, loss_dict, acc_dict = _network(tf_images, tf_shape, tf_labels)

    total_loss = frc.RPN_CLASSIFICATION_LOSS_WEIGHTS * loss_dict['rpn_cls_loss'] + \
                 frc.RPN_LOCATION_LOSS_WEIGHTS * loss_dict['rpn_bbox_loss'] + \
                 frc.FASTER_RCNN_CLASSIFICATION_LOSS_WEIGHTS * loss_dict['rcnn_cls_loss'] + \
                 frc.FASTER_RCNN_LOCATION_LOSS_WEIGHTS * loss_dict['rcnn_bbox_loss'] + \
                 0.0005 * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    global_step = slim.get_or_create_global_step()

    learning_rate = tf.train.exponential_decay(learning_rate=0.003, global_step=0, decay_steps=10, decay_rate=0.5)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    with tf.name_scope('summary'):
        tf.summary.scalar('loss/total_loss', total_loss)
        tf.summary.scalar('loss/rpn_cls_loss', loss_dict['rpn_cls_loss'])
        tf.summary.scalar('loss/rpn_bbox_loss', loss_dict['rpn_bbox_loss'])
        tf.summary.scalar('loss/rcnn_cls_loss', loss_dict['rcnn_cls_loss'])
        tf.summary.scalar('loss/rcnn_bbox_loss', loss_dict['rcnn_bbox_loss'])
        tf.summary.scalar('accuracy/rpn_acc',  acc_dict['rpn_cls_acc'])
        tf.summary.scalar('accuracy/rcnn_acc', acc_dict['rcnn_cls_acc'])

    summary_op = tf.summary.merge_all()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=frc.REFRESH_LOGS_ITERS)

    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(frc.SUMMARY_PATH, graph=sess.graph)

        for step in range(frc.MAXIMUM_ITERS):
            images, gt_bboxes = _image_batch(frc.IMAGE_SHAPE)
            feed_dict = {tf_images: images, tf_labels: gt_bboxes, tf_shape: frc.IMAGE_SHAPE}

            if step % frc.REFRESH_LOGS_ITERS != 0:
                _, global_step_ = sess.run([train_op, global_step], feed_dict)
            else:
                _, total_loss_, rpn_cls_loss_, rpn_bbox_loss_, rcnn_cls_loss_, rcnn_bbox_loss_, \
                rpn_cls_acc_, rcnn_cls_acc_, summary_str, global_step_ = \
                    sess.run([train_op, total_loss, loss_dict['rpn_cls_loss'], loss_dict['rpn_bbox_loss'],
                              loss_dict['rcnn_cls_loss'], loss_dict['rcnn_bbox_loss'],
                              acc_dict['rpn_cls_acc'], acc_dict['rcnn_cls_acc'], summary_op, global_step], feed_dict)

                print(f'Iter {step}: ',
                      f'total_loss: {total_loss_:.3}',
                      f'rpn_cls_loss: {rpn_cls_loss_:.3}',
                      f'rpn_bbox_loss: {rpn_bbox_loss_:.3}',
                      f'rcnn_cls_loss: {rcnn_cls_loss_:.3}',
                      f'rcnn_bbox_loss: {rcnn_bbox_loss_:.3}',
                      f'rpn_cls_acc: {rpn_cls_acc_:.3}',
                      f'rcnn_cls_acc: {rcnn_cls_acc_:.3}')

                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                saver.save(sess, frc.MODEL_SAVE_PATH)


if __name__ == '__main__':
    _main()
