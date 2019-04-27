import os
import sys
import time
from importlib import import_module

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from toy_dataset.shape_generator import generate_shape_image
from region_proposal_network_batch import rpn_batch
from faster_rcnn_batch import faster_rcnn, batchwise_process_faster_rcnn, build_faster_rcnn_losses

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
                           weights_regularizer=slim.l2_regularizer(frc.L2_WEIGHT),
                           scope='rpn_feature')

    # RPN
    rpn_cls_loss, rpn_cls_acc, rpn_bbox_loss, rois, labels, bbox_targets = rpn_batch(features, image_shape, gt_bboxes)

    # RCNN
    # Get cls_score in shape of [FASTER_RCNN_MINIBATCH_SIZE, CLS_NUM + 1]
    # Get bbox_pred in shape of [FASTER_RCNN_MINIBATCH_SIZE, 4 * (CLS_NUM + 1)]
    cls_score, bbox_pred = faster_rcnn(features, rois)

    cls_prob = slim.softmax(cls_score)
    cls_categories = tf.cast(tf.argmax(cls_prob, axis=1), dtype=tf.int32)
    rcnn_cls_acc = tf.reduce_mean(tf.cast(tf.equal(cls_categories, tf.cast(tf.reshape(labels, [-1]), tf.int32)), tf.float32))

    final_bbox_list, final_score_list, final_categories_list = batchwise_process_faster_rcnn(rois, bbox_pred, cls_prob, image_shape)

    rcnn_bbox_loss, rcnn_cls_loss = build_faster_rcnn_losses(bbox_pred, bbox_targets, cls_prob, labels, frc.NUM_CLS + 1)

    # ------------------------------BEGIN SUMMARY--------------------------------
    # Image summary for RPN rois
    class_names = frc.CLS_NAMES + ['circle', 'rectangle', 'triangle']
    display_rois_img = tf.reshape(inputs[0], shape=[frc.IMAGE_SHAPE[0], frc.IMAGE_SHAPE[1], 3])
    with tf.name_scope('rpn_image_summary'):
        display_BG_indices = tf.reshape(tf.where(tf.equal(labels[0], 0)), [-1])
        display_FG_indices = tf.reshape(tf.where(tf.not_equal(labels[0], 0)), [-1])

        display_BG_rois = tf.gather(rois[0], display_BG_indices)
        display_FG_rois = tf.gather(rois[0], display_FG_indices)

        display_BG_img = tf.py_func(draw_rectangle, [display_rois_img, display_BG_rois], [tf.uint8])
        display_FG_img = tf.py_func(draw_rectangle, [display_rois_img, display_FG_rois], [tf.uint8])

    tf.summary.image('class_rois/BG', display_BG_img)
    tf.summary.image('class_rois/FG', display_FG_img)

    # Add predicted bbox with confidence 0.25, 0.5, 0.75 and ground truth in image summary.
    with tf.name_scope('rcnn_image_summary'):
        final_bbox = final_bbox_list[0]
        final_score = final_score_list[0]
        final_categories = final_categories_list[0]
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

        show_gt = tf.reshape(tf.gather(gt_bboxes[:, 1:], tf.where(tf.equal(gt_bboxes[:, 0], 0))), [-1, 5])

        display_image_25 = tf.py_func(draw_rectangle_with_name,
                                      [display_rois_img, display_bboxes_25, display_categories_25, class_names],
                                      [tf.uint8])
        display_image_50 = tf.py_func(draw_rectangle_with_name,
                                      [display_rois_img, display_bboxes_50, display_categories_50, class_names],
                                      [tf.uint8])
        display_image_75 = tf.py_func(draw_rectangle_with_name,
                                      [display_rois_img, display_bboxes_75, display_categories_75, class_names],
                                      [tf.uint8])
        display_image_gt = tf.py_func(draw_rectangle_with_name,
                                      [display_rois_img, show_gt[:, :-1], show_gt[:, -1], class_names],
                                      [tf.uint8])

    tf.summary.image('detection/gt', display_image_gt)
    tf.summary.image('detection/25', display_image_25)
    tf.summary.image('detection/50', display_image_50)
    tf.summary.image('detection/75', display_image_75)
    # -------------------------------END SUMMARY---------------------------------

    loss_dict = {'rpn_cls_loss': rpn_cls_loss,
                 'rpn_bbox_loss': rpn_bbox_loss,
                 'rcnn_cls_loss': rcnn_cls_loss,
                 'rcnn_bbox_loss': rcnn_bbox_loss}
    acc_dict = {'rpn_cls_acc': rpn_cls_acc,
                'rcnn_cls_acc': rcnn_cls_acc}

    return final_bbox, final_score, final_categories, loss_dict, acc_dict


def _image_batch(image_shape, batch_size=2):
    while True:
        batch_image, batch_gt, batch_image_shape = [], [], []
        for i in range(batch_size):
            image, bboxes, labels, _ = generate_shape_image(image_shape)
            labels = np.reshape(labels, [-1, 1])
            batch_ind = i * np.ones_like(labels)
            gt_labels = np.hstack([batch_ind, bboxes, labels])
            batch_image.append(image)
            batch_gt.append(gt_labels)
            batch_image_shape.append(image_shape)
        batch_image = np.stack(batch_image, axis=0)
        batch_gt = np.vstack(batch_gt)
        batch_image_shape = np.vstack(batch_image_shape)
        yield batch_image, batch_gt, batch_image_shape


def _preprocess(inputs, image_shape=None):
    return inputs


def _main():
    # dataset = tf.data.Dataset.from_generator(lambda: _image_batch(frc.IMAGE_SHAPE), (tf.float32, tf.int32, tf.int32))
    # dataset = dataset.batch(frc.IMAGE_BATCH_SIZE)
    # iterator = dataset.make_one_shot_iterator()
    batch_generator = _image_batch(frc.IMAGE_SHAPE, frc.IMAGE_BATCH_SIZE)

    with tf.name_scope('inputs'):
        tf_images = tf.placeholder(dtype=tf.float32,
                                   shape=[None, frc.IMAGE_SHAPE[0], frc.IMAGE_SHAPE[1], 3],
                                   name='images')
        tf_labels = tf.placeholder(dtype=tf.int32, shape=[None, 6], name='ground_truth_bbox')
        tf_shape = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='image_shape')

    # Preprocess input images
    preprocessed_inputs = _preprocess(tf_images)

    final_bbox, final_score, final_categories, loss_dict, acc_dict = _network(preprocessed_inputs, tf_shape, tf_labels)

    total_loss = frc.RPN_CLASSIFICATION_LOSS_WEIGHTS * loss_dict['rpn_cls_loss'] + \
                 frc.RPN_LOCATION_LOSS_WEIGHTS * loss_dict['rpn_bbox_loss'] + \
                 frc.FASTER_RCNN_CLASSIFICATION_LOSS_WEIGHTS * loss_dict['rcnn_cls_loss'] + \
                 frc.FASTER_RCNN_LOCATION_LOSS_WEIGHTS * loss_dict['rcnn_bbox_loss'] + \
                 tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.train.piecewise_constant(global_step, frc.LEARNING_RATE_BOUNDARIES, frc.LEARNING_RATE_SCHEDULAR)

    # Adam
    # train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

    # Momentum
    train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(total_loss, global_step=global_step)

    # RMS
    # train_op = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9).minimize(total_loss, global_step=global_step)

    # Add train summary.
    with tf.name_scope('loss'):
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('rpn_cls_loss', loss_dict['rpn_cls_loss'])
        tf.summary.scalar('rpn_bbox_loss', loss_dict['rpn_bbox_loss'])
        tf.summary.scalar('rcnn_cls_loss', loss_dict['rcnn_cls_loss'])
        tf.summary.scalar('rcnn_bbox_loss', loss_dict['rcnn_bbox_loss'])
    with tf.name_scope('accuracy'):
        tf.summary.scalar('rpn_acc',  acc_dict['rpn_cls_acc'])
        tf.summary.scalar('rcnn_acc', acc_dict['rcnn_cls_acc'])
    with tf.name_scope('train'):
        tf.summary.scalar('learning_rate', learning_rate)

    summary_op = tf.summary.merge_all()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver()

    if not os.path.exists(frc.SUMMARY_PATH):
        os.mkdir(frc.SUMMARY_PATH)

    with tf.Session() as sess:
        if frc.PRE_TRAIN_MODEL_PATH:
            print('Load pre-trained model:', frc.PRE_TRAIN_MODEL_PATH)
            saver.restore(sess, frc.PRE_TRAIN_MODEL_PATH)
        else:
            sess.run(init_op)

        start_time = time.strftime('%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(frc.SUMMARY_PATH, start_time)
        save_model_dir = os.path.join(log_dir, 'model')

        if not os.path.exists(save_model_dir):
            os.mkdir(log_dir)
            os.mkdir(save_model_dir)
        summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)

        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess, coord)

        try:
            for step in range(frc.MAXIMUM_ITERS + 1):
                # batch_image, batch_gt, batch_labels = iterator.get_next()
                # batch_gt = tf.squeeze(batch_gt, axis=0)
                # batch_labels = tf.squeeze(batch_labels, axis=0)
                # images, gt_bboxes, image_shape = sess.run([batch_image, batch_gt, batch_labels])
                images, gt_bboxes, image_shape = batch_generator.__next__()
                feed_dict = {tf_images: images, tf_labels: gt_bboxes, tf_shape: image_shape}

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

                    saver.save(sess, os.path.join(save_model_dir, frc.MODEL_NAME + '.ckpt'), step)

        # except tf.errors.OutOfRangeError:
        #     print('done')
        except KeyboardInterrupt:
            print('Interrupt in epoch', step)
        finally:
            # coord.request_stop()
        # coord.join(threads)
            summary_writer.close()


if __name__ == '__main__':
    _main()
    # generator = _image_batch(frc.IMAGE_SHAPE, frc.IMAGE_BATCH_SIZE)
    # im, gt, sp = generator.__next__()
    # print(im.shape)
    # print(gt.shape)
    # print(sp.shape)