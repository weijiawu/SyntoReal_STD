import numpy as np
import cv2
from scipy.stats import mode, norm


def apply_canny(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)


class SWT(object):
    def __init__(self, image_file, boxes, direction='both+'):
        ## Read image
        self.image_file = image_file
        img = cv2.imread(image_file)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = rgb_img
        self.h, self.w = img.shape[:2]

        self.boxes = boxes

        self.direction = direction
        self.STEP_LIMIT = 10
        self.SWT_TOTAL_COUNT = 10
        self.SWT_STD_LIM = 20.0
        self.STROKE_WIDTH_SIZE_RATIO_LIM = 0.02  ## Min value
        self.STROKE_WIDTH_VARIANCE_RATIO_LIM = 0.25  ## Min value

        self.gray_img = cv2.cvtColor(self.img.copy(), cv2.COLOR_RGB2GRAY)  # 灰度图
        self.canny_img = apply_canny(self.img)

        self.sobelX = cv2.Sobel(self.gray_img, cv2.CV_64F, 1, 0, ksize=-1)  # 边缘检测
        self.sobelY = cv2.Sobel(self.gray_img, cv2.CV_64F, 0, 1, ksize=-1)

        self.stepsX = self.sobelY.astype(int)  ## Steps are inversed!! (x-step -> sobelY)
        self.stepsY = self.sobelX.astype(int)

        self.magnitudes = np.sqrt(self.stepsX * self.stepsX + self.stepsY * self.stepsY)
        self.gradsX = self.stepsX / (self.magnitudes + 1e-10)
        self.gradsY = self.stepsY / (self.magnitudes + 1e-10)

    def get_stroke_properties(self, stroke_widths):
        if len(stroke_widths) == 0:
            return (0, 0, 0, 0, 0, 0)
        try:
            most_probable_stroke_width = mode(stroke_widths, axis=None)[0][0]
            most_probable_stroke_width_count = mode(stroke_widths, axis=None)[1][0]
        except IndexError:
            most_probable_stroke_width = 0
            most_probable_stroke_width_count = 0
        try:
            mean, std = norm.fit(stroke_widths)
            x_min, x_max = int(min(stroke_widths)), int(max(stroke_widths))
        except ValueError:
            mean, std, x_min, x_max = 0, 0, 0, 0
        return most_probable_stroke_width, most_probable_stroke_width_count, mean, std, x_min, x_max

    def get_strokes(self, x, y, h, w, text_map):
        stroke_widths = np.array([[np.Infinity, np.Infinity]])

        for i in range(y, y + h):
            for j in range(x, x + w):
                if self.canny_img[i, j] != 0 and text_map[i, j] == 1:
                    gradX = self.gradsX[i, j]
                    gradY = self.gradsY[i, j]

                    prevX, prevY, prevX_opp, prevY_opp, step_size = i, j, i, j, 0

                    if self.direction == "light":
                        go, go_opp = True, False
                    elif self.direction == "dark":
                        go, go_opp = False, True
                    else:
                        go, go_opp = True, True

                    stroke_width = np.Infinity
                    stroke_width_opp = np.Infinity
                    while (go or go_opp) and (step_size < self.STEP_LIMIT):
                        step_size += 1
                        if go:
                            curX = np.int(np.floor(i + gradX * step_size))
                            curY = np.int(np.floor(j + gradY * step_size))
                            #                             print(curX)
                            if (curX <= y or curY <= x or curX >= y + h or curY >= x + w):
                                go = False
                            if go and ((curX != prevX) or (curY != prevY)):
                                try:
                                    if self.canny_img[curX, curY] != 0:
                                        if np.arccos(gradX * -self.gradsX[curX, curY] + gradY * -self.gradsY[
                                            curX, curY]) < np.pi / 2.0:
                                            stroke_width = int(np.sqrt((curX - i) ** 2 + (curY - j) ** 2))
                                            go = False
                                except IndexError:
                                    go = False

                                prevX = curX
                                prevY = curY

                        if go_opp:
                            curX_opp = np.int(np.floor(i - gradX * step_size))
                            curY_opp = np.int(np.floor(j - gradY * step_size))
                            if (curX_opp <= y or curY_opp <= x or curX_opp >= y + h or curY_opp >= x + w):
                                go_opp = False
                            if go_opp and ((curX_opp != prevX_opp) or (curY_opp != prevY_opp)):
                                try:
                                    if self.canny_img[curX_opp, curY_opp] != 0:
                                        if np.arccos(gradX * -self.gradsX[curX_opp, curY_opp] + gradY * -self.gradsY[
                                            curX_opp, curY_opp]) < np.pi / 2.0:
                                            stroke_width_opp = int(np.sqrt((curX_opp - i) ** 2 + (curY_opp - j) ** 2))
                                            go_opp = False

                                except IndexError:
                                    go_opp = False

                                prevX_opp = curX_opp
                                prevY_opp = curY_opp

                    stroke_widths = np.append(stroke_widths, [(stroke_width, stroke_width_opp)], axis=0)

        stroke_widths_opp = np.delete(stroke_widths[:, 1], np.where(stroke_widths[:, 1] == np.Infinity))
        stroke_widths = np.delete(stroke_widths[:, 0], np.where(stroke_widths[:, 0] == np.Infinity))
        return stroke_widths, stroke_widths_opp

    def detect(self):
        final_box = []
        for i, (bbox) in enumerate(self.boxes):
            # print("x, y, w, h",x, y, w, h)

            coordinate = np.around(bbox[:8].reshape((4, 2))).astype(np.int32)

            x_max = min(max(coordinate[:, 0]), self.w - 1)
            x_min = min(coordinate[:, 0])
            y_max = min(max(coordinate[:, 1]), self.h - 1)
            y_min = min(coordinate[:, 1])
            x, y, h, w = x_min, y_min, (y_max - y_min), (x_max - x_min)

            text_map = np.zeros((self.h, self.w))
            cv2.fillPoly(text_map, [coordinate], 1)

            stroke_widths, stroke_widths_opp = self.get_strokes(x, y, h, w, text_map)
            #             print(stroke_widths.shape,stroke_widths_opp.shape)
            stroke_widths = np.append(stroke_widths, stroke_widths_opp, axis=0)
            # std: 标准差,表示离散程度
            # stroke_width：most_probable_stroke_width
            stroke_width, stroke_width_count, _, std, _, _ = self.get_stroke_properties(stroke_widths)

            # print("stroke_widths:", stroke_widths.shape)
            # print("std:", std)
            if len(stroke_widths) < self.SWT_TOTAL_COUNT:
                continue
            if std > self.SWT_STD_LIM:
                continue

            #             stroke_width_size_ratio = stroke_width / max(w, h)
            #             print("stroke_width_size_ratio:",stroke_width_size_ratio)
            stroke_width_variance_ratio = stroke_width / (std * std + 1e-10)
            # print("stroke_width_variance_ratio:", stroke_width_variance_ratio)

            #             if stroke_width_size_ratio < self.STROKE_WIDTH_SIZE_RATIO_LIM:
            #                 continue
            if stroke_width_variance_ratio > self.STROKE_WIDTH_VARIANCE_RATIO_LIM:
                final_box.append(bbox)
        return np.array(final_box)
#             break


