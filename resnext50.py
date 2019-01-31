import tensorflow as tf
from tensorflow.contrib import slim


_bn_params = {'decay': 0.995, 'epsilon': 0.0001}
_l2_weight = 0.0005


def _preprocess(inputs, input_size):
    return inputs


def _conv2d_block(net, filters, block_num, conv_num, cardinality=32, projection=False):
    with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=_bn_params,
                        weights_regularizer=tf.nn.l2_normalize(_l2_weight)):
        cardinality_list = []
        for i in range(cardinality):
            if not projection:
                cardinality_net = slim.conv2d(net, filters // cardinality, [1, 1],
                                              scope='conv{}_{}_1_c{}'.format(block_num, conv_num, i))
            else:
                cardinality_net = slim.conv2d(net, filters // cardinality, [1, 1], 2,
                                              scope='conv{}_{}_1_c{}'.format(block_num, conv_num, i))
            cardinality_net = slim.conv2d(cardinality_net, filters // cardinality, [3, 3],
                                          scope='conv{}_{}_2_c{}'.format(block_num, conv_num, i))
            cardinality_net = slim.conv2d(cardinality_net, 2 * filters, [1, 1],
                                          activation_fn=None,
                                          scope='conv{}_{}_3_c{}'.format(block_num, conv_num, i))
            cardinality_list.append(cardinality_net)
        residul_net = sum(cardinality_list)

        return tf.nn.relu(net + residul_net, name='conv{}_relu'.format(block_num))


def inference(inputs, name='resnext50', global_average_pooling=False):
    # preprocess 224 x 224 x 3
    preprocessed_inputs = _preprocess(inputs)

    with tf.variable_scope(name, 'resnext50'):

        # conv1 224 x 224 x 3 => 112 x 112 x 64
        with tf.name_scope(name + '_conv1'):
            with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=_bn_params,
                                weights_regularizer=tf.nn.l2_normalize(_l2_weight)):
                net = slim.conv2d(preprocessed_inputs, 64, [7, 7], 2, padding='VALID', scope='conv1')
                net = slim.max_pool2d(net, 2, scope='pool1')

        # conv2 112 x 112 x 64 => 56 x 56 x 256
        with tf.name_scope(name + '_conv2'):
            for i in range(3):
                if i == 0:
                    net = _conv2d_block(net, 128, 2, i, projection=True)
                else:
                    net = _conv2d_block(net, 128, 2, i)

        # conv3 56 x 56 x 256 => 28 x 28 x 512
        with tf.name_scope(name + '_conv3'):
            for i in range(4):
                if i == 0:
                    net = _conv2d_block(net, 256, 3, i, projection=True)
                else:
                    net = _conv2d_block(net, 256, 3, i)

        # conv4 28 x 28 x 512 => 14 x 14 x 1024
        with tf.name_scope(name + '_conv4'):
            for i in range(6):
                if i == 0:
                    net = _conv2d_block(net, 512, 4, i, projection=True)
                else:
                    net = _conv2d_block(net, 512, 4, i)

        # conv5 14 x 14 x 1024 => 7 x 7 x 2048
        with tf.name_scope(name + '_conv5'):
            for i in range(3):
                if i == 0:
                    net = _conv2d_block(net, 1024, 5, i, projection=True)
                else:
                    net = _conv2d_block(net, 1024, 5, i)

        # global average pooling
        if global_average_pooling:
            with tf.name_scope(name + '_global_average_pooling'):
                net = slim.avg_pool2d(net, 2)

    return net
