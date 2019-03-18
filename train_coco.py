import os
import sys
import time
from importlib import import_module
import random

import skimage.io as io
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from toy_dataset.coco_dataset import load_translated_data, get_gt_infos
from region_proposal_network import rpn
from faster_rcnn import faster_rcnn, process_faster_rcnn, build_faster_rcnn_losses

from utils.image_draw import draw_rectangle_with_name, draw_rectangle
import faster_rcnn_configs as frc


def _network(inputs, image_shape, gt_bboxes, cls_names):
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
    image_shape = tf.cast(tf.reshape(image_shape, [-1]), dtype=tf.int32)
    gt_bboxes = tf.cast(tf.reshape(gt_bboxes, [-1, 5]), dtype=tf.int32)
    rpn_cls_loss, rpn_cls_acc, rpn_bbox_loss, rois, labels, bbox_targets = rpn(features, image_shape, gt_bboxes)

    # Image summary for RPN rois
    class_names = frc.CLS_NAMES + cls_names
    display_rois_img = inputs[0]
    display_bg_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])
    display_fg_indices = tf.reshape(tf.where(tf.not_equal(labels, 0)), [-1])
    display_bg_rois = tf.gather(rois, display_bg_indices)
    display_fg_rois = tf.gather(rois, display_fg_indices)
    display_bg_img = tf.py_func(draw_rectangle, [display_rois_img, display_bg_rois], [tf.uint8])
    display_fg_img = tf.py_func(draw_rectangle, [display_rois_img, display_fg_rois], [tf.uint8])
    rpn_image_bg_summary = tf.summary.image('class_rois/background', display_bg_img)
    rpn_image_fg_summary = tf.summary.image('class_rois/foreground', display_fg_img)

    # RCNN
    cls_score, bbox_pred = faster_rcnn(features, rois, image_shape)

    cls_prob = slim.softmax(cls_score)
    cls_categories = tf.cast(tf.argmax(cls_prob, axis=1), dtype=tf.int32)
    rcnn_cls_acc = tf.reduce_mean(tf.cast(tf.equal(cls_categories, tf.cast(labels, tf.int32)), tf.float32))

    final_bbox, final_score, final_categories = process_faster_rcnn(rois, bbox_pred, cls_prob, image_shape)

    rcnn_bbox_loss, rcnn_cls_loss = build_faster_rcnn_losses(bbox_pred, bbox_targets, cls_prob, labels, frc.NUM_CLS + 1)

    # ------------------------------BEGIN SUMMARY--------------------------------
    # Add predicted bbox with confidence 0.25, 0.5, 0.75 and ground truth in image summary.
    with tf.name_scope('rcnn_image_summary'):
        # display_indices_25 = tf.reshape(tf.where(tf.greater_equal(final_score, 0.25) &
        #                                          tf.less(final_score, 0.5) &
        #                                          tf.not_equal(final_categories, 0)), [-1])
        # display_indices_50 = tf.reshape(tf.where(tf.greater_equal(final_score, 0.5) &
        #                                          tf.less(final_score, 0.75) &
        #                                          tf.not_equal(final_categories, 0)), [-1])
        display_indices_75 = tf.reshape(tf.where(tf.greater_equal(final_score, 0.75) &
                                                 tf.not_equal(final_categories, 0)), [-1])

        # display_bboxes_25 = tf.gather(final_bbox, display_indices_25)
        # display_bboxes_50 = tf.gather(final_bbox, display_indices_50)
        display_bboxes_75 = tf.gather(final_bbox, display_indices_75)
        # display_categories_25 = tf.gather(final_categories, display_indices_25)
        # display_categories_50 = tf.gather(final_categories, display_indices_50)
        display_categories_75 = tf.gather(final_categories, display_indices_75)

        # display_image_25 = tf.py_func(draw_rectangle_with_name,
        #                               [inputs[0], display_bboxes_25, display_categories_25, class_names],
        #                               [tf.uint8])
        # display_image_50 = tf.py_func(draw_rectangle_with_name,
        #                               [inputs[0], display_bboxes_50, display_categories_50, class_names],
        #                               [tf.uint8])
        display_image_75 = tf.py_func(draw_rectangle_with_name,
                                      [inputs[0], display_bboxes_75, display_categories_75, class_names],
                                      [tf.uint8])
        display_image_gt = tf.py_func(draw_rectangle_with_name,
                                      [inputs[0], gt_bboxes[:, :-1], gt_bboxes[:, -1], class_names],
                                      [tf.uint8])

    rcnn_gt_image_summary = tf.summary.image('detection/gt', display_image_gt)
    # tf.summary.image('detection/25', display_image_25)
    # tf.summary.image('detection/50', display_image_50)
    rcnn_75_image_summary = tf.summary.image('detection/75', display_image_75)
    image_summary = tf.summary.merge([rpn_image_bg_summary, rpn_image_fg_summary,
                                           rcnn_75_image_summary, rcnn_gt_image_summary])
    # -------------------------------END SUMMARY---------------------------------

    loss_dict = {'rpn_cls_loss': rpn_cls_loss,
                 'rpn_bbox_loss': rpn_bbox_loss,
                 'rcnn_cls_loss': rcnn_cls_loss,
                 'rcnn_bbox_loss': rcnn_bbox_loss}
    acc_dict = {'rpn_cls_acc': rpn_cls_acc,
                'rcnn_cls_acc': rcnn_cls_acc}

    return final_bbox, final_score, final_categories, loss_dict, acc_dict, image_summary


def _image_batch(image_list, label_list, size_list, batch_size=1):
    total_samples = len(image_list)
    while True:
        ind = random.choice(range(total_samples))
        img = io.imread(image_list[ind])
        img_dims = len(img.shape)
        if img_dims == 2:
            img = np.dstack([img, img, img])
        try:
            img = img[np.newaxis, :, :, :]
        except IndexError as err:
            print('Image dimention:', img_dims)
            print('Image ID:', image_list[ind])
            raise err
        gt_bboxes = get_gt_infos(label_list[ind])
        gt_bboxes = np.array(gt_bboxes, dtype=np.int32)
        img_size = size_list[ind]
        # img_size = np.array(size_list[ind], dtype=np.int32)
        yield img, gt_bboxes, img_size


def _preprocess(inputs, gt_bboxes, image_size, minimum_length=1000, is_training=True):
    height, width = tf.to_float(image_size[0]), tf.to_float(image_size[1])
    x1, y1, x2, y2, cls = tf.unstack(gt_bboxes, axis=1)
    minimum_size = tf.minimum(height, width)

    rate = minimum_length / minimum_size

    true_fn = lambda: (minimum_length, tf.to_int32(tf.round(width * rate)))
    false_fn = lambda: (tf.to_int32(tf.round(height * rate)), minimum_length)
    new_height, new_width = tf.cond(tf.equal(minimum_size, height), true_fn, false_fn)

    outputs = tf.image.resize_bilinear(inputs, size=(new_height, new_width))
    new_x1 = tf.to_int32(tf.to_float(x1) * rate)
    new_y1 = tf.to_int32(tf.to_float(y1) * rate)
    new_x2 = tf.to_int32(tf.to_float(x2) * rate)
    new_y2 = tf.to_int32(tf.to_float(y2) * rate)

    return outputs, tf.stack([new_x1, new_y1, new_x2, new_y2, cls], axis=1), \
           tf.stack([new_height, new_width], axis=0)


def _main():
    train_file_list, train_label_list, train_image_size_list, \
    val_file_list, val_label_list, val_image_size_list, cls_names = load_translated_data(
        '/media/wx/新加卷/datasets/COCODataset')

    batch_generator = _image_batch(train_file_list, train_label_list, train_image_size_list)

    with tf.name_scope('inputs'):
        tf_images = tf.placeholder(dtype=tf.float32,
                                   shape=[frc.IMAGE_BATCH_SIZE, None, None, 3],
                                   name='images')
        tf_labels = tf.placeholder(dtype=tf.int32, shape=[None, 5], name='ground_truth_bbox')
        tf_shape = tf.placeholder(dtype=tf.int32, shape=[None], name='image_shape')

    # Preprocess input images
    preprocessed_inputs, preprocessed_labels, preprocessed_shape = _preprocess(tf_images, tf_labels, tf_shape)

    final_bbox, final_score, final_categories, loss_dict, acc_dict, image_summary = _network(preprocessed_inputs,
                                                                                             preprocessed_shape,
                                                                                             preprocessed_labels,
                                                                                             cls_names)

    total_loss = frc.RPN_CLASSIFICATION_LOSS_WEIGHTS * loss_dict['rpn_cls_loss'] + \
                 frc.RPN_LOCATION_LOSS_WEIGHTS * loss_dict['rpn_bbox_loss'] + \
                 frc.FASTER_RCNN_CLASSIFICATION_LOSS_WEIGHTS * loss_dict['rcnn_cls_loss'] + \
                 frc.FASTER_RCNN_LOCATION_LOSS_WEIGHTS * loss_dict['rcnn_bbox_loss'] + \
                 tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.train.piecewise_constant(global_step, frc.LEARNING_RATE_BOUNDARIES, frc.LEARNING_RATE_SCHEDULAR)

    # Adam
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
    # train_op = tf.train.AdamOptimizer(0.003).minimize(total_loss, global_step=global_step)

    # Momentum
    # train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(total_loss, global_step=global_step)

    # RMS
    # train_op = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9).minimize(total_loss, global_step=global_step)

    # Add train summary.
    with tf.name_scope('loss'):
        total_loss_summary = tf.summary.scalar('total_loss', total_loss)
        rpn_cls_loss_summary = tf.summary.scalar('rpn_cls_loss', loss_dict['rpn_cls_loss'])
        rpn_bbox_loss_summary = tf.summary.scalar('rpn_bbox_loss', loss_dict['rpn_bbox_loss'])
        rcnn_cls_loss_summary = tf.summary.scalar('rcnn_cls_loss', loss_dict['rcnn_cls_loss'])
        rcnn_bbox_loss_summary = tf.summary.scalar('rcnn_bbox_loss', loss_dict['rcnn_bbox_loss'])
    with tf.name_scope('accuracy'):
        rpn_cls_acc_summary = tf.summary.scalar('rpn_acc',  acc_dict['rpn_cls_acc'])
        rcnn_cls_acc_summary = tf.summary.scalar('rcnn_acc', acc_dict['rcnn_cls_acc'])
    with tf.name_scope('train'):
        lr_summary = tf.summary.scalar('learning_rate', learning_rate)

    # summary_op = tf.summary.merge_all()
    scale_summary = tf.summary.merge([total_loss_summary, rpn_cls_loss_summary, rpn_bbox_loss_summary,
                                      rcnn_cls_loss_summary, rcnn_bbox_loss_summary,
                                      rpn_cls_acc_summary, rcnn_cls_acc_summary, lr_summary])
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=4)

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

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        step = 0

        try:
            while step < frc.MAXIMUM_ITERS + 1:
                images, gt_bboxes, image_shape = batch_generator.__next__()
                if len(gt_bboxes) == 0:
                    continue

                feed_dict = {tf_images: images, tf_labels: gt_bboxes, tf_shape: image_shape}

                step_time = time.time()

                _, total_loss_, rpn_cls_loss_, rpn_bbox_loss_, rcnn_cls_loss_, rcnn_bbox_loss_, \
                rpn_cls_acc_, rcnn_cls_acc_, scale_summary_str, image_summary_str, global_step_ = \
                    sess.run([train_op, total_loss, loss_dict['rpn_cls_loss'], loss_dict['rpn_bbox_loss'],
                              loss_dict['rcnn_cls_loss'], loss_dict['rcnn_bbox_loss'],
                              acc_dict['rpn_cls_acc'], acc_dict['rcnn_cls_acc'],
                              scale_summary, image_summary,
                              global_step], feed_dict)

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

                if step % frc.REFRESH_LOGS_ITERS == 0 and step != 0:
                    summary_writer.add_summary(scale_summary_str, step)
                    saver.save(sess, os.path.join(save_model_dir, frc.MODEL_NAME + '.ckpt'), step)
                    if step % 100 == 0:
                        summary_writer.add_summary(image_summary_str)

                summary_writer.flush()
                step += 1

                # if step % frc.REFRESH_LOGS_ITERS != 0:
                #     _, global_step_ = sess.run([train_op, global_step], feed_dict)
                # else:
                #     step_time = time.time()
                #
                #     _, total_loss_, rpn_cls_loss_, rpn_bbox_loss_, rcnn_cls_loss_, rcnn_bbox_loss_, \
                #     rpn_cls_acc_, rcnn_cls_acc_, summary_str, global_step_ = \
                #         sess.run([train_op, total_loss, loss_dict['rpn_cls_loss'], loss_dict['rpn_bbox_loss'],
                #                   loss_dict['rcnn_cls_loss'], loss_dict['rcnn_bbox_loss'],
                #                   acc_dict['rpn_cls_acc'], acc_dict['rcnn_cls_acc'], summary_op, global_step], feed_dict)
                #
                #     step_time = time.time() - step_time
                #
                #     print(f'Iter: {step}',
                #           f'| total_loss: {total_loss_:.3}',
                #           f'| rpn_cls_loss: {rpn_cls_loss_:.3}',
                #           f'| rpn_bbox_loss: {rpn_bbox_loss_:.3}',
                #           f'| rcnn_cls_loss: {rcnn_cls_loss_:.3}',
                #           f'| rcnn_bbox_loss: {rcnn_bbox_loss_:.3}',
                #           f'| rpn_cls_acc: {rpn_cls_acc_:.3}',
                #           f'| rcnn_cls_acc: {rcnn_cls_acc_:.3}',
                #           f'| time: {step_time:.3}s')
                #
                #     summary_writer.add_summary(summary_str, step)
                #     summary_writer.flush()
                #
                #     saver.save(sess, os.path.join(save_model_dir, frc.MODEL_NAME + '.ckpt'), step)
                #     step += 1

        except tf.errors.OutOfRangeError:
            print('done')
        finally:
            coord.request_stop()
        coord.join(threads)
    summary_writer.close()


if __name__ == '__main__':
    _main()
