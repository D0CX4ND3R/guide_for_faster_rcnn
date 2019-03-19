import tensorflow as tf
from tensorflow.contrib import slim


_bn_params = {'decay': 0.995, 'epsilon': 0.0001}
_l2_weight = 0.0005


def _conv2d_block(net, filters, block_num, conv_num, projection=False, is_trining=True):
    with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=_bn_params,
                        weights_regularizer=slim.l2_regularizer(_l2_weight),
                        trainable=is_trining):

        if projection:
            residul = slim.conv2d(net, filters, [1, 1], 2, scope='conv{}_{}_1'.format(block_num, conv_num))
        else:
            residul = slim.conv2d(net, filters, [1, 1], scope='conv{}_{}_1'.format(block_num, conv_num))
        residul = slim.conv2d(residul, filters, [3, 3], scope='conv{}_{}_2'.format(block_num, conv_num))
        residul = slim.conv2d(residul, 2 * filters, [1, 1], activation_fn=None,
                              scope='conv{}_{}_3'.format(block_num, conv_num))

        if projection:
            net = slim.conv2d(net, 2 * filters, [3, 3], 2, activation_fn=None, normalizer_fn=slim.batch_norm,
                              normalizer_params=_bn_params, weights_regularizer=slim.l2_regularizer(_l2_weight),
                              scope='conv{}_branch'.format(block_num))

        return tf.nn.relu(net + residul, name='conv{}_relu'.format(block_num))


def inference(inputs, is_training=True, name='resnet50'):
    with tf.variable_scope(name, 'resnet50'):

        # conv1 224 x 224 x 3 => 112 x 112 x 64
        with tf.variable_scope(name + '_conv1'):
            with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=_bn_params,
                                weights_regularizer=slim.l2_regularizer(_l2_weight),
                                trainable=is_training):
                net = slim.conv2d(inputs, 64, [7, 7], 2, padding='SAME', scope='conv1')
                net = slim.max_pool2d(net, 2, scope='pool1')

        # conv2 112 x 112 x 64 => 56 x 56 x 256
        with tf.variable_scope(name + '_conv2'):
            for i in range(3):
                if i == 0:
                    net = _conv2d_block(net, 128, 2, i, projection=True)
                else:
                    net = _conv2d_block(net, 128, 2, i)

        # conv3 56 x 56 x 256 => 28 x 28 x 512
        with tf.variable_scope(name + '_conv3'):
            for i in range(4):
                if i == 0:
                    net = _conv2d_block(net, 256, 3, i, projection=True)
                else:
                    net = _conv2d_block(net, 256, 3, i)

        # conv4 28 x 28 x 512 => 14 x 14 x 1024
        with tf.variable_scope(name + '_conv4'):
            for i in range(6):
                if i == 0:
                    net = _conv2d_block(net, 512, 4, i, projection=True)
                else:
                    net = _conv2d_block(net, 512, 4, i)
    return net


def head(net):
    with tf.variable_scope('resnet50', reuse=tf.AUTO_REUSE):
        # conv5 14 x 14 x 1024 => 7 x 7 x 2048
        with tf.variable_scope('resnet50_conv5'):
            for i in range(3):
                if i == 0:
                    net = _conv2d_block(net, 1024, 5, i, projection=True)
                else:
                    net = _conv2d_block(net, 1024, 5, i)

        # global average pooling
        net = tf.reduce_mean(net, axis=[1, 2], name='global_average_pooling')
    return net