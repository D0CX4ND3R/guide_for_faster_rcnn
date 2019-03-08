import numpy as np
import cv2
import time


def generate_shape_image(image_size, n=4):
    img_h, img_w = image_size
    image = np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)
    # image = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    centers, radius = _gen_centers(n, image_size)
    bboxes = []
    labels = []
    areas = []
    draw_data = []
    for i in range(n):
        rect, area, label, data = _gen_shape(image_size, centers[i], radius[i])
        bboxes.append(rect)
        areas.append(area)
        labels.append(label)
        draw_data.append(data)

    sorted_index = np.argsort(areas)[::-1]
    bboxes = np.array(bboxes)
    labels = np.array(labels)
    draw_data = np.array(draw_data)

    bboxes = bboxes[sorted_index]
    labels = labels[sorted_index]
    areas = np.sort(areas)[::-1]
    draw_data = draw_data[sorted_index]

    for i in range(len(labels)):
        data = draw_data[i]
        color = _random_color()
        if labels[i] == 1:
            image = cv2.circle(image, (data[0][0], data[0][1]), data[1], color, -1)
        elif labels[i] == 2:
            # print(data[0], data[1])
            image = cv2.rectangle(image, (data[0][0], data[0][1]), (data[1][0], data[1][1]), color, -1)
        else:
            image = cv2.fillConvexPoly(image, data, color)
    return image, bboxes, labels, areas


def _calc_box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _gen_centers(n, image_size, board_rate=0.2):
    # np.random.seed(round(time.time()) % 13)
    img_h, img_w = image_size
    board_w = np.round(board_rate * img_w)
    board_h = np.round(board_rate * img_h)

    unit1 = np.round(np.sqrt(n))
    unit2 = np.round(n / unit1)
    xx, yy = np.meshgrid(np.arange(unit1), np.arange(unit2))
    x_step = img_w // unit1
    y_step = img_h // unit2
    radius = np.int32(np.round(np.minimum(x_step, y_step) / np.random.randint(2, 6, n)))
    center_pos = []
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            pt = [np.random.randint(x_step * xx[i, j] + board_w, x_step * (xx[i, j] + 1) - board_w),
                  np.random.randint(y_step * yy[i, j] + board_h, y_step * (yy[i, j] + 1) - board_h)]
            center_pos.append(pt)
    # print(center_pos)
    center_pos = np.array(center_pos)
    # print(center_pos)

    return center_pos, np.sort(radius)[::-1]


def _gen_shape(image_shape, center_pos, radius, shape_type=None, offset=15):
    if shape_type is None:
        shape_type = np.random.randint(1, 4)

    img_h, img_w = image_shape

    if shape_type == 1:
        rect = [np.maximum(0, int(center_pos[0] - radius - offset)),
                np.maximum(0, int(center_pos[1] - radius - offset)),
                np.minimum(img_w, int(center_pos[0] + radius + offset)),
                np.minimum(img_h, int(center_pos[1] + radius + offset))]
        area = np.pi * radius ** 2
        data = [center_pos, radius]
    elif shape_type == 2:
        w = radius * 1.5
        h = (np.random.rand() + 0.5) * w
        pt1 = (int(np.maximum(0, center_pos[0] - w // 2)),
               int(np.minimum(img_w, center_pos[1] - h // 2)))
        pt2 = (int(np.maximum(0, center_pos[0] + w // 2)),
               int(np.minimum(img_h, center_pos[1] + h // 2)))
        rect = [np.maximum(0, pt1[0] - offset), np.maximum(0, pt1[1] - offset),
                np.minimum(img_w, pt2[0] + offset), np.minimum(img_h, pt2[1] + offset)]
        area = _calc_box_area(rect)
        data = [pt1, pt2]
    else:
        data, rect, area = _gen_triangle(center_pos, radius)
        rect[:2] = np.maximum(0, rect[:2] - offset)
        rect[2:] = np.minimum([img_w, img_h], rect[2:] + offset)

    return rect, area, shape_type, data


def _random_color(colors=None):
    if colors is not None:
        assert type(colors) == list
        delta = np.mean(np.vstack(colors), axis=0) - 127.5
        color = np.zeros(3)
        color[delta < 0] = 255
    else:
        color = np.random.randint(50, 255, (3,))

    return int(color[0]), int(color[1]), int(color[2])


def _gen_triangle(center_pos, radius):
    angle = np.random.randint(70, 150, (2,))
    vec1 = np.random.rand(2)
    vec1 = vec1 / np.linalg.norm(vec1)
    pt1 = center_pos + radius * vec1
    cos_angle0 = np.cos(np.deg2rad(angle[0]))
    sin_angle0 = np.sin(np.deg2rad(angle[0]))
    cos_angle1 = np.cos(np.deg2rad(np.sum(angle)))
    sin_angle1 = np.sin(np.deg2rad(np.sum(angle)))

    rotate_mat0 = np.array([[cos_angle0, -sin_angle0], [sin_angle0, cos_angle0]])
    rotate_mat1 = np.array([[cos_angle1, -sin_angle1], [sin_angle1, cos_angle1]])

    vec2 = np.matmul(rotate_mat0, vec1)
    vec3 = np.matmul(rotate_mat1, vec1)

    pt2 = center_pos + radius * vec2
    pt3 = center_pos + radius * vec3

    pts = np.int32(np.vstack([pt1, pt2, pt3]))

    rect = np.hstack([np.min(pts, axis=0), np.max(pts, axis=0)])

    area = 0.5 * (np.sin(np.deg2rad(angle[0])) + np.sin(np.deg2rad(angle[1])) +
                  np.sin(np.deg2rad(360 - np.sum(angle)))) * radius ** 2

    return pts, rect, area


if __name__ == '__main__':
    cls_names = ['BG', 'circle', 'rectangle', 'triangle']


    while cv2.waitKey(1500) & 0xFF != ord('q'):
        im, bboxes, labels, areas = generate_shape_image((448, 448))

        num_target = len(labels)
        for i in range(num_target):
            box = bboxes[i]
            l = labels[i]
            info = '{}: {} \t{:.3}'.format(i, cls_names[l], float(areas[i]))
            print(info)
            im = cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)
            im = cv2.putText(im, cls_names[l] + str(areas[i]), (box[0] + 5, box[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            # cv2.imshow(info, im.copy())
        cv2.imshow('image', im)
    # cv2.waitKey()
    cv2.destroyAllWindows()
