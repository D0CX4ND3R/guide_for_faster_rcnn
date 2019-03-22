import os
import random
import sys
import time
from importlib import import_module

import numpy as np
import skimage.io as io
import cv2
import tensorflow as tf
from tensorflow.contrib import slim

import faster_rcnn_configs as frc
from toy_dataset.coco_dataset import get_gt_infos, load_translated_data


LEARNING_RATE_BOUNDARIES = [1000, 2000, 8000]
LEARNING_RATE_SCHEDULAR = [0.1, 0.01, 0.001, 0.0001]


def _batch_generator(image_list, label_list, batch_size=128, target_shape=(224, 224)):
    total_samples = len(image_list)
    image_batch = np.zeros(shape=(batch_size, target_shape[0], target_shape[1], 3))
    label_batch = np.zeros(shape=(batch_size, ), dtype=np.int32)
    batch = 0
    while True:
        ind = random.choice(range(total_samples))
        img = io.imread(image_list[ind])
        img_dims = len(img.shape)
        if img_dims == 2:
            img = np.dstack([img, img, img])

        gt_bboxes = get_gt_infos(label_list[ind])
        for gt_bbox in gt_bboxes:
            x1, y1, x2, y2, cls = gt_bbox
            sub_img = img[y1:y2+1, x1:x2+1, :]
            sub_img = cv2.resize(sub_img, target_shape)
            flip = random.choice([None, -1, 0, 1])
            if flip:
                sub_img = cv2.flip(sub_img, flip)
            image_batch[batch] = np.float32(sub_img)
            label_batch[batch] = cls - 1
            batch += 1
            if batch == batch_size - 1:
                batch = 0
                yield image_batch, label_batch


def _preprocess_images(inputs):
    outputs = inputs - tf.constant(frc.MEAN_COLOR)
    return outputs


def _network(inputs, tf_labels):
    if 'backbones' not in sys.path:
        sys.path.append('backbones')
    backbone = import_module(frc.BACKBONE)

    net = backbone.inference(inputs)
    features = slim.conv2d(net, 512, [3, 3], normalizer_fn=slim.batch_norm,
                           normalizer_params={'decay': 0.995, 'epsilon': 0.0001},
                           weights_regularizer=slim.l2_regularizer(frc.L2_WEIGHT),
                           scope='rpn_feature')
    with tf.variable_scope('rcnn'):
        net = backbone.head(features)

    flatten_net = slim.flatten(net, scope='flatten')
    logits = slim.fully_connected(flatten_net, frc.NUM_CLS, activation_fn=None,
                                  normalizer_fn=slim.batch_norm, normalizer_params={'decay': 0.995, 'epsilon': 0.0001},
                                  weights_regularizer=slim.l2_regularizer(frc.L2_WEIGHT), scope='fc')
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf_labels))
    pred = tf.cast(tf.argmax(slim.softmax(logits), axis=1), dtype=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, tf_labels), dtype=tf.float32))
    return cross_entropy, acc


def _main():
    train_file_list, train_label_list, train_image_size_list, \
    val_file_list, val_label_list, val_image_size_list, cls_names = load_translated_data(
        '/media/wx/新加卷/datasets/COCODataset')

    with tf.name_scope('inputs'):
        tf_image = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='images')
        tf_label = tf.placeholder(dtype=tf.int32, shape=[None], name='labels')

    cross_entropy, acc = _network(tf_image, tf_label)
    reg_loss = 0.001 * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    total_loss = cross_entropy + reg_loss

    global_step = tf.train.get_or_create_global_step()

    # learning_rate = tf.train.piecewise_constant(global_step, LEARNING_RATE_BOUNDARIES, LEARNING_RATE_SCHEDULAR)

    # Adam
    optimizer = tf.train.AdamOptimizer(0.003)

    # Momentum
    # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)

    train_op = optimizer.minimize(total_loss, global_step=global_step)

    with tf.name_scope('summary'):
        tf.summary.scalar('loss', total_loss)
        tf.summary.scalar('cross_entropy', cross_entropy)
        tf.summary.scalar('reg_loss', reg_loss)
        tf.summary.scalar('accuracy', acc)

    summary_op = tf.summary.merge_all()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    all_variables = slim.get_model_variables()
    saved_variables = [var for var in all_variables
                       if var.op.name.split('/')[-1] == 'weights' and var.op.name.split('/')[0] != 'fc']

    saver = tf.train.Saver(var_list=saved_variables, max_to_keep=4)

    batch_gen = _batch_generator(train_file_list, train_label_list)

    log_dir = '/home/wx/source_code/PycharmProjects/count_rebars/logs'
    with tf.Session() as sess:
        start_time = time.strftime('%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(log_dir, start_time + '_pretrain')
        model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
            os.mkdir(model_dir)
        summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)

        sess.run([init_op])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        step = 0
        try:
            while step <= 50000:
                step_time = time.time()
                images, labels = batch_gen.__next__()
                feed_in = {tf_image: images, tf_label: labels}
                _, loss_, xent_, reg_loss_, acc_, sry_str_, global_step_ = sess.run(
                    [train_op, total_loss, cross_entropy, reg_loss, acc, summary_op, global_step], feed_dict=feed_in)

                if step % 10 == 0:
                    summary_writer.add_summary(sry_str_, step)
                if step % 100 == 0:
                    saver.save(sess, os.path.join(model_dir, frc.BACKBONE + '_pretrain.ckpt'), step)

                step_time = time.time() - step_time
                print(f'Epoch: {step} |',
                      f'loss: {loss_:.3} |',
                      f'cross entropy: {xent_:.3} |',
                      f'reg loss: {reg_loss_:.3} |',
                      f'accuracy: {acc_:.3} |',
                      f'time: {step_time:.3}')

                step += 1
        except tf.errors.OutOfRangeError:
            print('Done!')
        finally:
            coord.request_stop()
        coord.join(threads)
        summary_writer.close()


if __name__ == '__main__':
    _main()