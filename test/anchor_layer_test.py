import numpy as np
import cv2


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


height = width = 28
feat_stride = 8

shift_x = np.arange(height) * feat_stride
shift_y = np.arange(width) * feat_stride

shift_x, shift_y = np.meshgrid(shift_x, shift_y)
shifts = np.vstack([shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()]).transpose()

im1 = np.zeros((height * feat_stride, width * feat_stride, 3), dtype=np.uint8)
im2 = np.zeros((height * feat_stride, width * feat_stride, 3), dtype=np.uint8)

im1[shifts[:, 1], shifts[:, 0], 2] = 255
im2[shifts[:, 1], shifts[:, 0], 0] = 255

cv2.namedWindow('1', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('2', cv2.WINDOW_AUTOSIZE)

cv2.imshow('1', im1)
cv2.imshow('2', im2)
key = cv2.waitKey() & 0xFF
while key != ord('q'):
    continue
cv2.destroyAllWindows()
