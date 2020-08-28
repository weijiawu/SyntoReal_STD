import numpy as np
from shapely.geometry import Polygon
import numpy as np
import cv2
from PIL import Image
import math
import os
import torch


def get_MSER(image_path,):
    image = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    negtive_map = np.zeros(image.shape[:2])
    mser = cv2.MSER_create(_delta=5, _min_area=10, _max_variation=0.8)
    regions, bboxes = mser.detectRegions(rgb_img)
    # 绘制文本区域（不规则轮廓）
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    keep = []
    for hull in hulls:
        x, y, w, h = cv2.boundingRect(hull)
        keep.append([x, y, x + w, y + h])

    # 使用非极大值抑制获取不重复的矩形框
    pick = non_max_suppression_fast(np.array(keep), overlapThresh=0.4)

    # loop over the picked bounding boxes and draw them
    for (startX, startY, endX, endY) in pick:
        cv2.fillPoly(negtive_map, np.array([[[startX, startY], [endX, startY], [endX, endY], [startX, endY]]]), (1))
    negtive_map = Image.fromarray(negtive_map)

    return negtive_map


def crop_img_target_source(img, vertices, labels, length,negtive_map):
    '''crop img patches to obtain batch and augment
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        length      : length of cropped image region
    Output:
        region      : cropped image region
        new_vertices: new vertices in cropped region
    '''
    h, w = img.height, img.width
    # confirm the shortest side of image >= length
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
        negtive_map = negtive_map.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)
        negtive_map = negtive_map.resize((length, int(h * length / w)), Image.BILINEAR)

    ratio_w = img.width / w
    ratio_h = img.height / h
    assert (ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:, [0, 2, 4, 6]] = vertices[:, [0, 2, 4, 6]] * ratio_w
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * ratio_h

    # find random position
    remain_h = img.height - length
    remain_w = img.width - length
    flag = True
    cnt = 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text([start_w, start_h], length, new_vertices[labels == 1, :])
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    negtive_map = negtive_map.crop(box)

    if new_vertices.size == 0:
        return region, new_vertices,negtive_map

    new_vertices[:, [0, 2, 4, 6]] -= start_w
    new_vertices[:, [1, 3, 5, 7]] -= start_h
    return region, new_vertices,negtive_map


def rotate_img_target_source(img, vertices, negtive_map,angle_range=10):
    '''rotate image [-10, 10] degree to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        angle_range : rotate range
    Output:
        img         : rotated PIL Image
        new_vertices: rotated vertices
    '''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    negtive_map = negtive_map.rotate(angle, Image.BILINEAR)

    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i, :] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x], [center_y]]))
    return img, new_vertices,negtive_map

def adjust_height_target_source(img, vertices, negtive_map, ratio=0.2):
    '''adjust height of image to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        ratio       : height changes in [0.8, 1.2]
    Output:
        img         : adjusted PIL Image
        new_vertices: adjusted vertices
    '''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)  #[0.8, 1.2]
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)
    negtive_map = negtive_map.resize((img.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * (new_h / old_h)
    return img, new_vertices,negtive_map

def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''get rotated locations of all pixels for next stages
    Input:
        rotate_mat: rotatation matrix
        anchor_x  : fixed x position
        anchor_y  : fixed y position
        length    : length of image
    Output:
        rotated_x : rotated x positions <numpy.ndarray, (length,length)>
        rotated_y : rotated y positions <numpy.ndarray, (length,length)>
    '''
    x = np.arange(length)
    y = np.arange(length)
    x, y = np.meshgrid(x, y)
    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, x.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                    np.array([[anchor_x], [anchor_y]])
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)
    return rotated_x, rotated_y


def get_score_geo_target_source(img, vertices, labels, scale, length, negative_map):
    '''generate score gt and geometry gt
    Input:
        img     : PIL Image
        vertices: vertices of text regions <numpy.ndarray, (n,8)>
        labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        scale   : feature map / image                                    0.25
        length  : image length                                           512
    Output:
        score gt, geo gt, ignored
    '''
    score_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
    geo_map = np.zeros((int(img.height * scale), int(img.width * scale), 5), np.float32)
    ignored_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)


    negative_map = np.array(negative_map)
    negative_map = cv2.resize(negative_map,(int(img.height * scale), int(img.width * scale)))
    negative_map = np.expand_dims(negative_map, -1)

    index = np.arange(0, length, int(1 / scale)) # [0   4   8  12  ....  508]
    index_x, index_y = np.meshgrid(index, index) # Return coordinate matrices from coordinate vectors.
    ignored_polys = []
    polys = []

    for i, vertice in enumerate(vertices):

        ignored_polys.append(np.around(scale * vertice.reshape((4, 2))).astype(np.int32))

        #得到0.3缩放后的score map   将vectice缩小到原来的0.3
        poly = np.around(scale * shrink_poly(vertice).reshape((4, 2))).astype(np.int32)  # scaled & shrinked
        polys.append(poly)
        temp_mask = np.zeros(score_map.shape[:-1], np.float32)
        cv2.fillPoly(temp_mask, [poly], 1)

        theta = find_min_rect_angle(vertice)
        rotate_mat = get_rotate_mat(theta)

        rotated_vertices = rotate_vertices(vertice, theta)
        x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
        rotated_x, rotated_y = rotate_all_pixels(rotate_mat, vertice[0], vertice[1], length)

        # print("rotated_x：",rotated_x.shape)
        # print("rotated_y：", rotated_y.shape)
        d1 = rotated_y - y_min
        d1[d1 < 0] = 0
        d2 = y_max - rotated_y
        d2[d2 < 0] = 0
        d3 = rotated_x - x_min
        d3[d3 < 0] = 0
        d4 = x_max - rotated_x
        d4[d4 < 0] = 0
        geo_map[:, :, 0] += d1[index_y, index_x] * temp_mask
        geo_map[:, :, 1] += d2[index_y, index_x] * temp_mask
        geo_map[:, :, 2] += d3[index_y, index_x] * temp_mask
        geo_map[:, :, 3] += d4[index_y, index_x] * temp_mask
        geo_map[:, :, 4] += theta * temp_mask

    cv2.fillPoly(ignored_map, ignored_polys, 1)

    # negative_map: 背景
    # ignored_map：前景
    text_background = negative_map*ignored_map
    negative_map = negative_map*(1-text_background)
    ignored_map = ignored_map*(1-text_background)

    ignored_map = 1 - (negative_map + ignored_map)
    cv2.fillPoly(score_map, polys, 1)
    return torch.Tensor(score_map).permute(2, 0, 1), torch.Tensor(geo_map).permute(2, 0, 1), torch.Tensor(
        ignored_map).permute(2, 0, 1)


def non_max_suppression_fast(boxes, overlapThresh):
    """
    boxes: boxes为一个m*n的矩阵，m为bbox的个数，n的前4列为每个bbox的坐标，
           格式为（x1,y1,x2,y2），有时会有第5列，该列为每一类的置信
    overlapThresh: 最大允许重叠率
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of all bounding boxes respectively
    x1 = boxes[:,0]  # startX
    y1 = boxes[:,1]  # startY
    x2 = boxes[:,2]  # endX
    y2 = boxes[:,3]  # endY
    # probs = boxes[:,4]

    # compute the area of the bounding boxes and sort the bboxes
    # by the bottom y-coordinate of the bboxes by ascending order
    # and grab the indexes of the sorted coordinates of bboxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # if probabilities are provided, sort by them instead
    # idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the idxs list
    while len(idxs) > 0:
        # grab the last index in the idxs list (the bottom-right box)
        # and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest coordinates for the start of the bbox
        # and the smallest coordinates for the end of the bbox
        # in the rest of bounding boxes.
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # the ratio of overlap in the bounding box
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that overlap is larger than overlapThresh
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")



def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def move_points(vertices, index1, index2, r, coef):
    '''move the two points to shrink edge
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        index1  : offset of point1
        index2  : offset of point2
        r       : [r1, r2, r3, r4] in paper
        coef    : shrink ratio in paper
    Output:
        vertices: vertices where one edge has been shinked
    '''
    index1 = index1 % 4
    index2 = index2 % 4
    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    return vertices


def shrink_poly(vertices, coef=0.3):
    '''shrink the text region
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        coef    : shrink ratio in paper
    Output:
        v       : vertices of shrinked text region <numpy.ndarray, (8,)>
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1, y1, x2, y2), cal_distance(x1, y1, x4, y4))
    r2 = min(cal_distance(x2, y2, x1, y1), cal_distance(x2, y2, x3, y3))
    r3 = min(cal_distance(x3, y3, x2, y2), cal_distance(x3, y3, x4, y4))
    r4 = min(cal_distance(x4, y4, x1, y1), cal_distance(x4, y4, x3, y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if cal_distance(x1, y1, x2, y2) + cal_distance(x3, y3, x4, y4) > \
            cal_distance(x2, y2, x3, y3) + cal_distance(x1, y1, x4, y4):
        offset = 0  # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1  # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v


def get_rotate_mat(theta):
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):
    '''rotate vertices around anchor
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4, 2)).T

    # print(vertices)
    # print("v:",v.shape)
    # 取第一个顶点
    if anchor is None:
        anchor = v[:, :1]
    # print(anchor)

    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)


def get_boundary(vertices):
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max


def cal_error(vertices):
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err


def find_min_rect_angle(vertices):
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
    angle_interval = 1
    angle_list = list(range(-90, 90, angle_interval))
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


def is_cross_text(start_loc, length, vertices):
    '''check if the crop image crosses text regions
    Input:
        start_loc: left-top position
        length   : length of crop image
        vertices : vertices of text regions <numpy.ndarray, (n,8)>
    Output:
        True if crop image crosses text region
    '''
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    a = np.array([start_w, start_h, start_w + length, start_h, \
                  start_w + length, start_h + length, start_w, start_h + length]).reshape((4, 2))
    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4, 2))).convex_hull
        inter = p1.intersection(p2).area
        try:
            if 0.01 <= inter / p2.area <= 0.99:
                return True
        except:
            continue
    return False