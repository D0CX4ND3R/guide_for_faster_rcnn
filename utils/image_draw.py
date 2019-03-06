import numpy as np
import cv2


def draw_rectangle_with_name(image, bboxes, categories, cls_names):
    image = np.uint8(image)
    n = len(bboxes)
    for i in range(n):
        box = bboxes[i]
        cls = cls_names[int(categories[i])]
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))

        image = cv2.rectangle(image, pt1, pt2, (0, 255, 0), 1)
        # image = putText(image, cls, (pt1[0], pt1[1]), 0, 1, (0, 255, 0), 1)

    return image


def draw_rectangle(image, bboxes):
    image = np.uint8(image)
    n = len(bboxes)
    for i in range(n):
        box = bboxes[i]
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))

        image = cv2.rectangle(image, pt1, pt2, (0, 255, 0), 1)

    return image


def draw_rectangle2(image, bboxes):
    image = np.uint8(image)
    n = len(bboxes)
    for i in range(n):
        box = bboxes[i]
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))

        image = cv2.rectangle(image, pt1, pt2, (0, 0, 255), 1)

    return image


if __name__ == '__main__':
    img = np.zeros((448, 448, 3), dtype=np.float32)

    cls_names = ['BG', 'circle', 'rectangle', 'triangle']
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
