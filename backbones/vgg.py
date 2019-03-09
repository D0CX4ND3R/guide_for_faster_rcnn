import tensorflow as tf
from tensorflow.contrib import slim


_bn_params = {'decay': 0.995, 'epsilon': 0.0001}
_l2_weight = 0.0005

# STRIDE_SIZE = 16


def inference(inputs, is_training=True, num_layers=11, name='vgg'):
    assert type(num_layers) == int
    assert num_layers in [11, 13, 16, 19]

    name = name + str(num_layers)
    with tf.variable_scope(name):
        with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.leaky_relu,
                            normalizer_fn=slim.batch_norm, normalizer_params=_bn_params,
                            weights_regularizer=slim.l2_regularizer(_l2_weight), trainable=is_training):
            # 224 x 224 x 3 => 112 x 112 x 64
            if num_layers == 11:
                net = slim.conv2d(inputs, 64, [3, 3], scope=name + '_conv1')
            else:
                net = slim.repeat(inputs, 2, slim.conv2d, num_outputs=64, kernel_size=[3, 3], scope=name + '_conv1')
            net = slim.max_pool2d(net, [2, 2], scope=name + '_pool1')

            # 112 x 112 x 64 => 56 x 56 x 128
            if num_layers == 11:
                net = slim.conv2d(net, 128, [3, 3], scope=name + '_conv2')
            else:
                net = slim.repeat(net, 2, slim.conv2d, num_outputs=128, kernel_size=[3, 3], scope=name + '_conv2')
            net = slim.max_pool2d(net, [2, 2], scope=name + '_pool2')

            # 56 x 56 x 128 => 28 x 28 x 256
            if num_layers < 16:
                net = slim.repeat(net, 2, slim.conv2d, num_outputs=256, kernel_size=[3, 3], scope=name + '_conv3')
            elif num_layers == 16:
                net = slim.repeat(net, 3, slim.conv2d, num_outputs=256, kernel_size=[3, 3], scope=name + '_conv3')
            else:
                net = slim.repeat(net, 4, slim.conv2d, num_outputs=256, kernel_size=[3, 3], scope=name + '_conv3')
            net = slim.max_pool2d(net, [2, 2], scope=name + '_pool3')

            # 28 x 28 x 256 => 14 x 14 x 512
            if num_layers < 16:
                net = slim.repeat(net, 2, slim.conv2d, num_outputs=512, kernel_size=[3, 3], scope=name + '_conv4')
            elif num_layers == 16:
                net = slim.repeat(net, 3, slim.conv2d, num_outputs=512, kernel_size=[3, 3], scope=name + '_conv4')
            else:
                net = slim.repeat(net, 4, slim.conv2d, num_outputs=512, kernel_size=[3, 3], scope=name + '_conv4')
            net = slim.max_pool2d(net, [2, 2], scope=name + '_pool4')

    return net


def head(net, feature_dim=1024, is_training=True, num_layers=11, name='vgg'):
    assert type(num_layers) == int
    assert num_layers in [11, 13, 16, 19]

    name = name + str(num_layers)
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.leaky_relu,
                            weights_regularizer=slim.l2_regularizer(_l2_weight), trainable=is_training):
            with slim.arg_scope([slim.conv2d], padding='SAME',
                                normalizer_fn=slim.batch_norm, normalizer_params=_bn_params):
                # 14 x 14 x 512 => 7 x 7 x 512
                if num_layers < 16:
                    net = slim.repeat(net, 2, slim.conv2d, num_outputs=512, kernel_size=[3, 3], scope=name + '_conv5')
                elif num_layers == 16:
                    net = slim.repeat(net, 3, slim.conv2d, num_outputs=512, kernel_size=[3, 3], scope=name + '_conv5')
                else:
                    net = slim.repeat(net, 4, slim.conv2d, num_outputs=512, kernel_size=[3, 3], scope=name + '_conv5')
                net = slim.max_pool2d(net, [2, 2], scope=name + '_pool5')

                # net = slim.flatten(net, scope=name + '_flatten')
                # net = slim.fully_connected(net, 4096, scope=name + '_fc6')
                # net = slim.dropout(net, 0.8, is_training=is_training)
                # net = slim.fully_connected(net, 4096, scope=name + '_fc7')
                # net = slim.dropout(net, 0.8, is_training=is_training)
                # net = slim.fully_connected(net, feature_dim, activation_fn=None, scope=name + '_fc8')
                # net = slim.repeat(net, 2, slim.conv2d, num_outputs=feature_dim, kernel_size=[1, 1])
        net = tf.reduce_mean(net, axis=[1, 2], name='global_average_pooling')
    return net
