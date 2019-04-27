import numpy as np
import cv2


_colors = [(255, 255, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255),
           (192, 192, 192), (192, 192, 0), (0, 192, 192), (192, 0, 192), (192, 0, 0), (0, 192, 0), (0, 0, 192),
           (128, 128, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128), (128, 0, 0), (0, 128, 0), (0, 0, 128),
           (64, 64, 64), (64, 64, 0), (0, 64, 64), (64, 0, 64), (64, 0, 0), (0, 64, 0), (0, 0, 64),
           (255, 255, 192), (192, 255, 255), (255, 192, 255), (255, 192, 192), (192, 255, 192), (192, 192, 255),
           (255, 255, 128), (128, 255, 255), (255, 128, 255), (255, 128, 128), (128, 255, 128), (128, 128, 255),
           (255, 255, 64), (64, 255, 255), (255, 64, 255), (255, 64, 64), (64, 255, 64), (64, 64, 255),
           (192, 192, 128), (128, 192, 192), (192, 128, 192), (192, 128, 128), (128, 192, 128), (128, 128, 192),
           (192, 192, 64), (64, 192, 192), (192, 64, 192), (192, 64, 64), (64, 192, 64), (64, 64, 192),
           (128, 128, 64), (64, 128, 128), (128, 64, 128), (128, 64, 64), (64, 128, 64), (64, 64, 128),
           (255, 192, 128), (255, 192, 64), (255, 192, 0), (255, 128, 192), (255, 128, 64), (255, 128, 0),
           (255, 64, 192), (255, 64, 128), (255, 64, 0), (255, 0, 192), (255, 0, 128), (255, 0, 64),
           (192, 255, 128), (192, 255, 64), (192, 255, 0), (192, 128, 255), (192, 128, 64), (192, 128, 0),
           (192, 64, 255), (192, 64, 128), (192, 64, 0), (192, 0, 255), (192, 0, 128), (192, 0, 64),
           (128, 255, 192), (128, 255, 64), (128, 255, 0), (128, 192, 255), (128, 192, 64), (128, 192, 0),
           (128, 64, 255), (128, 64, 192), (128, 64, 0), (128, 0, 255), (128, 0, 192), (128, 0, 64),
           (64, 255, 192), (64, 255, 128), (64, 255, 0), (64, 192, 255), (64, 192, 128), (64, 192, 0),
           (64, 128, 255), (64, 128, 192), (64, 128, 0), (64, 0, 255), (64, 0, 192), (64, 0, 128),
           (0, 255, 192), (0, 255, 128), (0, 255, 64), (0, 192, 255), (0, 192, 128), (0, 192, 64),
           (0, 128, 255), (0, 128, 192), (0, 128, 64), (0, 64, 255), (0, 64, 192), (0, 64, 128), (0, 0, 0)]


def draw_rectangle_with_name(image, bboxes, categories, cls_names, zoom=512):
    assert len(cls_names) <= len(_colors)
    img = np.uint8(image.copy())
    img_h, img_w = img.shape[0], img.shape[1]
    min_shape = np.minimum(img_h, img_w)

    thickness = int(np.round(3 / 448 * min_shape))
    font_thickness = int(np.round(2 / 448 * min_shape))
    font_scale = float(1.0 / 448 * min_shape)

    n = len(bboxes)

    for i in range(n):
        box = bboxes[i]
        categ = int(categories[i])
        cls = str(cls_names[categ])
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        pt = (int(box[0]), int(box[3]))

        img = cv2.rectangle(img, pt1, pt2, _colors[categ - 1], thickness)
        img = cv2.putText(img, cls, pt, cv2.FONT_HERSHEY_COMPLEX, font_scale, _colors[categ - 1], font_thickness)

    rate = min_shape / zoom
    if min_shape == img_h:
        new_h = zoom
        new_w = int(np.ceil(img_w / rate))
    else:
        new_w = zoom
        new_h = int(np.ceil(img_h / rate))

    img = cv2.resize(img, (new_w, new_h))

    return img


def draw_rectangle(image, bboxes, zoom=512):
    img = np.uint8(image.copy())
    img_h, img_w = img.shape[0], img.shape[1]
    min_shape = np.minimum(img_h, img_w)

    thickness = int(np.round(3 / 448 * min_shape))

    n = len(bboxes)
    for i in range(n):
        box = bboxes[i]
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))

        img = cv2.rectangle(img, pt1, pt2, (255, 0, 0), thickness)

    rate = min_shape / zoom
    if min_shape == img_h:
        new_h = zoom
        new_w = int(np.ceil(img_w / rate))
    else:
        new_w = zoom
        new_h = int(np.ceil(img_h / rate))

    img = cv2.resize(img, (new_w, new_h))

    return img


# def draw_rectangle2(image, bboxes):
#     image = np.uint8(image)
#     n = len(bboxes)
#     for i in range(n):
#         box = bboxes[i]
#         pt1 = (int(box[0]), int(box[1]))
#         pt2 = (int(box[2]), int(box[3]))
#
#         image = cv2.rectangle(image, pt1, pt2, (255, 0, 0), 1)

    # return image


if __name__ == '__main__':
    img = np.zeros((448, 448, 3), dtype=np.float32)

    cls_names = [u'BG', u'circle', u'rectangle', u'triangle']
    cate = [1, 2, 3]
    bboxes = np.array([[224.5 - 64.5, 224.5 - 64.5, 224.5 + 64.5, 224.5 + 64.5],
                       [224.5 - 128.5, 224.5 - 32.5, 224.5 + 128.5, 224.5 + 32.5],
                       [224.5 - 32.5, 224.5 - 128.5, 224.5 + 32.5, 224.5 + 128.5]])

    img_ = draw_rectangle_with_name(img, bboxes, cate, cls_names)

    cv2.imshow('test', img_)
    key = cv2.waitKey() & 0xFF
    cv2.destroyAllWindows()


# from PIL import Image, ImageDraw, ImageFont
#
#
# def draw_rectangle_with_name(image, box, cls_name):
