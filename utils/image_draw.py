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


def draw_rectangle_with_name(image, bboxes, categories, cls_names):
    assert len(cls_names) <= len(_colors)

    img = np.uint8(image.copy())
    img_h = img.shape[1]
    n = len(bboxes)

    for i in range(n):
        box = bboxes[i]
        categ = int(categories[i])
        cls = str(cls_names[categ])
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        pt = (int(np.maximum(0, box[0])),
              int(np.minimum(img_h, box[3])))

        img = cv2.rectangle(img, pt1, pt2, _colors[categ - 1], 3)
        img = cv2.putText(img, cls, pt, cv2.FONT_HERSHEY_COMPLEX, 1.0, _colors[categ - 1], 2)

    return img


def draw_rectangle(image, bboxes):
    image = np.uint8(image)
    n = len(bboxes)
    for i in range(n):
        box = bboxes[i]
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))

        image = cv2.rectangle(image, pt1, pt2, (255, 0, 0), 2)

    return image


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
