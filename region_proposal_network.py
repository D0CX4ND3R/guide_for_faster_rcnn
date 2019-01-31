import tensorflow as tf
from tensorflow.contrib import slim

import numpy as np


class RpnConfig(object):
    def __init__(self, class_count):
        self.scale = [0.5, 1, 2]
        self.size = [128, 256, 512]
        self.cls_count = class_count

    @property
    def anchor_num(self):
        return len(self.scale) * len(self.size)

    @property
    def class_count(self):
        return self.cls_count


def _cls():
    pass


def _reg():
    pass


def generate_anchors(original_anchor, scales=[8, 16, 32], ratios=[0.5, 1, 2]):
    def _gen_anchors(width, height):
        wn = np.repeat(width, len(scales)) * np.array(scales)
        hn = np.repeat(height, len(scales)) * np.array(scales)

        wn = wn[:, np.newaxis]
        hn = hn[:, np.newaxis]
        return np.hstack([x_center - (wn - 1) / 2, y_center - (hn - 1) / 2,
                          x_center + (wn - 1) / 2, y_center + (hn - 1) / 2])

    # calculate original anchor's width, height and center coordinate
    w = original_anchor[2] - original_anchor[0] + 1
    h = original_anchor[3] - original_anchor[1] + 1
    x_center = original_anchor[0] + (w - 1) / 2
    y_center = original_anchor[1] + (h - 1) / 2

    # calculate the original anchor's area
    original_area = w * h

    # calculate the three ratios areas
    three_ratios_area = original_area * np.array(ratios)

    # calculate the three kinds of areas width and height
    three_ratios_width = np.round(np.sqrt(three_ratios_area))
    three_ratios_height = three_ratios_width / np.array(ratios)

    # calculate anchors, each anchors coordinate is [x1, y1, x2, y2]
    # (x1, y1)-----------------------------------------
    # |                                                |
    # |                                                |
    # |                                                |
    # |                                                |
    # |                                                |
    # ------------------------------------------(x2, y2)
    anchors = [_gen_anchors(wr, hr) for wr, hr in zip(three_ratios_width, three_ratios_height)]

    return np.vstack(anchors)


def rpn(feature_map, rpn_config=RpnConfig(2)):
    with tf.variable_scope('rpn'):
        features = slim.conv2d(feature_map, 512, [3, 3], normalizer_fn=slim.batch_norm,
                               normalizer_params={'decacy': 0.995, 'epsilon': 0.0001},
                               weights_regularizer=tf.nn.l2_normalize(0.0005),
                               scope='rpn_feature')

        # rpn_cls_score
        rpn_cls_score = slim.conv2d(features, 2 * rpn_config.anchor_num, [1, 1],
                                    activation_fn=None, scope='rpn_cls_score')

        # rpn_cls_prob
        rpn_cls_prob = slim.conv2d(rpn_cls_score, rpn_config.anchor_num * rpn_config.class_count, [1, 1],
                                   activation_fn=None, scope='rpn_cls_prob')

if __name__ == '__main__':
    a = generate_anchors([0, 0, 15, 15])
    import matplotlib.pyplot as plt


    def draw_rectangle(ax, anchor, color):
        x1 = anchor[0]
        y1 = anchor[1]
        x2 = anchor[2]
        y2 = anchor[3]
        ax.hlines(y1, x1, x2, colors=color)
        ax.hlines(y2, x1, x2, colors=color)
        ax.vlines(x1, y1, y2, colors=color)
        ax.vlines(x2, y1, y2, colors=color)


    colors = ['b', 'g', 'r'] * 3
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for anchor, color in zip(a, colors):
        draw_rectangle(ax, anchor, color)
        x_ctr = (anchor[0] + anchor[2]) / 2
        y_ctr = (anchor[1] + anchor[3]) / 2
        print('center:')
        print('x=', x_ctr, 'y=', y_ctr)
        print('area:', (anchor[2] - anchor[0] + 1) * (anchor[3] - anchor[1] + 1))
        print('w=', anchor[2] - anchor[0] + 1, 'h=', anchor[3] - anchor[1] + 1)
        ax.scatter(x_ctr, y_ctr, c=color)
    ax.set_aspect('equal')
    plt.show()

