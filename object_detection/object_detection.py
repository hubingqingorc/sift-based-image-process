import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import time
import bisect


class ShowImg:  # 图片显示
    def __init__(self):
        self.figsize = (10, 8)
        self.dip = 120
        self.boundary_color = (0, 255, 0)
        self.boundary_width = 8
        self.corner_line_color = (0, 0, 255)
        self.corner_line_width = 8
        self.line_color = (255, 0, 0)
        self.line_width = 2
        self.save_filename = './object_detection_result.jpg'

    @staticmethod
    def bgr2rgb(img):  # 通道转换BGR2RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def rgb2bgr(img):  # 通道转换BGR2RGB
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def show_2_img(self, a, b):  # 显示两张图片
        plt.figure(figsize=self.figsize, dpi=self.dip)
        plt.subplot(121)
        plt.imshow(self.bgr2rgb(a))
        plt.subplot(122)
        plt.imshow(self.bgr2rgb(b))
        plt.show()

    @staticmethod
    def perspective_translation_point(matrix, point, translation_length):
        src_p = np.float32([[[point[0], point[1]]]])
        dst_p = cv2.perspectiveTransform(src_p, matrix)
        return int(dst_p[0][0][0] + translation_length), int(dst_p[0][0][1])

    def show_bounding_box_and_mapping_line(self, a, b, matrix, conrespond_table, inner_point_index):
        a_width, a_high, a_chl = a.shape
        b_width, b_high, b_chl = b.shape
        ax_x, ax_y = max(b_width, a_width), b_high + a_high
        background = np.zeros((ax_x, ax_y, 3)).astype(a.dtype)
        background[0:a_width, 0:a_high, :] = self.bgr2rgb(a)
        background[0:b_width, a_high:ax_y, :] = self.bgr2rgb(b)
        #  获取映射后的角点
        corner_0, corner_1, corner_2, corner_3 = (0, 0), (0, a_width), (a_high, 0), (a_high, a_width)
        corner_0_p = self.perspective_translation_point(matrix, corner_0, a_high)
        corner_1_p = self.perspective_translation_point(matrix, corner_1, a_high)
        corner_2_p = self.perspective_translation_point(matrix, corner_2, a_high)
        corner_3_p = self.perspective_translation_point(matrix, corner_3, a_high)
        #  绘制投影框
        cv2.line(background, corner_0_p, corner_1_p, self.boundary_color, self.boundary_width)
        cv2.line(background, corner_1_p, corner_3_p, self.boundary_color, self.boundary_width)
        cv2.line(background, corner_3_p, corner_2_p, self.boundary_color, self.boundary_width)
        cv2.line(background, corner_2_p, corner_0_p, self.boundary_color, self.boundary_width)
        #  绘制投影前后角点连接线
        cv2.line(background, corner_0, corner_0_p, self.corner_line_color, self.corner_line_width)
        cv2.line(background, corner_1, corner_1_p, self.corner_line_color, self.corner_line_width)
        cv2.line(background, corner_2, corner_2_p, self.corner_line_color, self.corner_line_width)
        cv2.line(background, corner_3, corner_3_p, self.corner_line_color, self.corner_line_width)
        #  绘制对应特征点连线
        for _i in range(len(inner_point_index)):
            cv2.line(background, (int(conrespond_table[inner_point_index[_i], 1]),
                                  int(conrespond_table[inner_point_index[_i], 2])),
                     (int(conrespond_table[inner_point_index[_i], 4]) + a_high,
                      int(conrespond_table[inner_point_index[_i], 5])), self.line_color, self.line_width)
        plt.imshow(background)
        plt.show()
        return background

    def save_img(self, img):
        cv2.imwrite(self.save_filename, self.rgb2bgr(img))


class SIFT:  # SIFT计算
    def __init__(self):  # 初始化算子
        self.sift = cv2.xfeatures2d.SIFT_create()

    def detect_compute(self, img):  # 计算SIFT关键点
        return self.sift.detectAndCompute(img, None)

    @staticmethod
    def get_sift_image(img, kp):  # 生成SIFT关键点显示图
        return cv2.drawKeypoints(img, kp, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


class RANSAC:  # RANSAC计算类
    def __init__(self, max_iter, inner_ratio, threshold, kp1, des1, kp2, des2):
        self.max_iter = max_iter  # 最大循环数
        self.inner_ratio = inner_ratio  # 内点率
        self.threshold = threshold  # 内点判断阈值
        self.kp1, self.des1, self.kp2, self.des2 = kp1, des1, kp2, des2
        self.kp1_index, self.kp2_index = range(len(kp1)), range(len(kp2))
        self.good_kp_threshold = 0.8  # 高质量尺度不变关键点L2距离比值
        self.sort_column_index = 40  # 近邻排序所使用的列，用单一列结合下面的阈值在获取图片关键点映射表有
        # 速度提升，但关键点映射数量和质量对应下降，使匹配质量下降
        self.neighbor_threshold = 10  # 近邻排序相对于输入点的上下限偏移

    def get_good_nearest_sift_kp(self, single_kp1_index, column_sorted, sorted_des2):  # 获得高质量的尺度不变关键点
        """
        引入高质量尺度不变点判断，如最小方差较次小方差小 self.good_kp_threshold 倍，则图2中点为图1中点的高质量尺度不变关键点
        :return index 图2中关键点位置值， good_label标明是否为高质量关键点:
        """
        sorted_index = 0  # 关键点顺序值
        good_label = 0
        diff_neighbor, left_index = self.neighbor_diff(single_kp1_index, column_sorted, sorted_des2)
        if len(diff_neighbor) > 1:
            diff_neighbor_sort = np.sort(diff_neighbor)
            minimum, min_2nd = diff_neighbor_sort[0], diff_neighbor_sort[1]
            if minimum / min_2nd < self.good_kp_threshold:
                good_label = 1
                sorted_index = np.argsort(diff_neighbor)[0] + left_index
        return [sorted_index, good_label]

    def neighbor_diff(self, single_kp1_index, column_sorted, sorted_des2):  # 获得近邻并返回L2距离
        left_index = bisect.bisect_left(column_sorted, self.des1[single_kp1_index][self.sort_column_index]
                                        - self.neighbor_threshold)
        right_index = bisect.bisect_right(column_sorted, self.des1[single_kp1_index][self.sort_column_index]
                                          + self.neighbor_threshold)
        return [np.sum(np.square(sorted_des2[left_index:right_index] - self.des1[single_kp1_index]), axis=1),
                left_index] if left_index != right_index else [[], 0]

    def get_conrespond_table(self):  # 获得图1中各点对应图2中点的映射表
        conrespond_table = []
        sort_column = self.des2[:, self.sort_column_index]
        sort_column_index = np.argsort(sort_column)
        column_sorted = sort_column[sort_column_index]
        sorted_des2 = self.des2[sort_column_index]
        for _i in self.kp1_index:
            _sorted_index, _good_label = self.get_good_nearest_sift_kp(_i, column_sorted, sorted_des2)
            if _good_label == 1:
                _index = sort_column_index[_sorted_index]
                conrespond_table.append([_i, self.kp1[_i].pt[0], self.kp1[_i].pt[1],
                                         _index, self.kp2[_index].pt[0], self.kp2[_index].pt[1]])
        return conrespond_table

    @staticmethod
    def get_correspond_4point(conrespond_table):  # 获取与随机点对应的4个点
        src_4p, dst_4p = [], []
        while len(src_4p) < 4:
            rdm_src_p = random.sample(range(len(conrespond_table)), 1)
            have = 0
            for _i in src_4p:  # 确保选择的4个点中没有重合点
                if conrespond_table[rdm_src_p[0]][1] == _i[0] and conrespond_table[rdm_src_p[0]][2] == _i[1]:
                    have = 1
                    break
            if have == 0:
                src_4p.append([conrespond_table[rdm_src_p[0]][1], conrespond_table[rdm_src_p[0]][2]])
                dst_4p.append([conrespond_table[rdm_src_p[0]][4], conrespond_table[rdm_src_p[0]][5]])
        return src_4p, dst_4p

    def get_homography(self, conrespond_table):  # 得到单应性矩阵
        src, dst = self.get_correspond_4point(conrespond_table)
        return cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))

    def count_inner_point(self, conrespond_table):  # 计算内点数量
        matrix = self.get_homography(conrespond_table)
        # 右乘投影矩阵得到初步的投影点
        ori_perspective = np.dot(np.column_stack((conrespond_table[:, 1:3], np.ones(len(conrespond_table)))), matrix.T)
        perspective = (ori_perspective.T / ori_perspective[:, 2]).T  # 将偏置项归一得到真实的投影点
        result = np.where(np.square(perspective[:, 0] - conrespond_table[:, 4]) +  # 得到满足内点阈值的结果
                          np.square(perspective[:, 1] - conrespond_table[:, 5]) < self.threshold, 1, 0)
        return np.sum(result), matrix, np.argwhere(result == 1).flatten()

    def compare_refresh(self):  # 比较内点数量，刷新内点数量和单应性矩阵
        start = time.time()
        conrespond_table = np.array(self.get_conrespond_table())
        print("calculate conrespond_table cost{}  len(table):{}".format(time.time() - start, len(conrespond_table)))
        inner_count = 0
        matrix = []
        inner_point_index = []
        for _i in range(self.max_iter):
            # print("iteration",_i)
            new_inner_count, new_matrix, new_inner_point_index = self.count_inner_point(conrespond_table)
            if new_inner_count > inner_count:
                inner_count = new_inner_count
                matrix = new_matrix
                inner_point_index = new_inner_point_index
                # ShowImg.show_bounding_box_and_mapping_line(img1, img2, np.array(matrix), conrespond_table,
                #                                            inner_point_index)
            print("iteration: {}  inner_count: {}".format(_i, inner_count))
            print("iteration: {}  cost_time: {}".format(_i, time.time() - start))
            if 1.0 * inner_count / len(conrespond_table) > self.inner_ratio:
                break
        print("inner count:", inner_count)
        print("T:\n", matrix)
        ShowImg.save_img(ShowImg.show_bounding_box_and_mapping_line(img1, img2, np.array(matrix), conrespond_table,
                                                                    inner_point_index))
        return inner_count, matrix


if __name__ == '__main__':
    start = time.time()
    img1, img2 = cv2.imread('./object_detection_a.jpg'), cv2.imread('./object_detection_b.jpg')  # 读取源图片
    ShowImg = ShowImg()  # 显示图片
    ShowImg.show_2_img(img1, img2)  # 显示源图
    sift_img = SIFT()  # 计算SIFT
    kp1, des1 = sift_img.detect_compute(img1)
    kp2, des2 = sift_img.detect_compute(img2)
    ransac = RANSAC(10000, 0.5, 1, kp1, des1, kp2, des2)
    inner_count, T = ransac.compare_refresh()
