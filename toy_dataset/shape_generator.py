import numpy as np
import cv2


def generate_shape_image(image_size, n=5):
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
        if labels[i] == 0:
            image = cv2.circle(image, (data[0][0], data[0][1]), data[1], color, -1)
        elif labels[i] == 1:
            image = cv2.rectangle(image, data[0], data[1], color, -1)
        else:
            image = cv2.fillConvexPoly(image, data, color)
    return image, bboxes, labels, areas


def _calc_box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _gen_centers(n, image_size, board_rate=0.2):
    img_h, img_w = image_size
    board_w = np.round(board_rate * img_w)
    board_h = np.round(board_rate * img_h)

    center_pos = np.hstack([np.random.randint(board_w, img_w - board_w, (n, 1)),
                            np.random.randint(board_h, img_h - board_h, (n, 1))])
    radius = np.int32(np.round(np.minimum(img_h, img_w) / np.random.randint(3, 8, n)))

    return center_pos, np.sort(radius)[::-1]


def _gen_shape(image_shape, center_pos, radius, shape_type=None):
    if shape_type is None:
        shape_type = np.random.randint(0, 3)

    img_h, img_w = image_shape

    if shape_type == 0:
        rect = [np.maximum(0, int(center_pos[0] - radius)),
                np.maximum(0, int(center_pos[1] - radius)),
                np.minimum(img_w, int(center_pos[0] + radius)),
                np.minimum(img_h, int(center_pos[1] + radius))]
        area = np.pi * radius ** 2
        data = [center_pos, radius]
    elif shape_type == 1:
        w = radius
        h = (np.random.rand() + 0.5) * w
        pt1 = (int(center_pos[0] - w // 2), int(center_pos[1] - h // 2))
        pt2 = (int(center_pos[0] + w // 2), int(center_pos[1] + h // 2))
        rect = [pt1[0], pt1[1], pt2[0], pt2[1]]
        area = _calc_box_area(rect)
        data = [pt1, pt2]
    else:
        data, rect, area = _gen_triangle(center_pos, radius)
        rect[:2] = np.maximum(0, rect[:2])
        rect[2:] = np.minimum([img_w, img_h], rect[2:])

    return rect, area, shape_type, data


def _random_color(colors=None):
    if colors is not None:
        assert type(colors) == list
        delta = np.mean(np.vstack(colors), axis=0) - 127.5
        color = np.zeros(3)
        color[delta < 0] = 255
    else:
        color = np.random.randint(0, 255, (3,))

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
    cls_names = ['circle', 'rectangle', 'triangle']
    im, bboxes, labels, areas = generate_shape_image((896, 896))

    num_target = len(labels)

    for i in range(num_target):
        box = bboxes[i]
        l = labels[i]
        info = '{}: {} \t{:.3}'.format(i, cls_names[l], areas[i])
        print(info)
        im = cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), 2)
        im = cv2.putText(im, cls_names[l] + str(areas[i]), (box[0] + 5, box[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1)
        # cv2.imshow(info, im.copy())
    cv2.imshow('image', im)
    cv2.waitKey()
    cv2.destroyAllWindows()
