import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import time


class ShowImg:  # 进行图片初始显示及拼接后显
    def __init__(self):  # 初始化
        self.figsize = (10, 8)
        self.dip = 120
        self.boundary_color = (0, 255, 0)
        self.boundary_width = 8
        self.save_filename = './object_tracking_try.jpg'

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
    def get_perspective_point(matrix, point):
        src_p = np.float32([[[point[0], point[1]]]])
        dst_p = cv2.perspectiveTransform(src_p, matrix)
        return dst_p[0][0][0], dst_p[0][0][1]

    def add_bounding_box(self, a_shape, b, matrix):
        a_width, a_high, a_chl = a_shape
        corner_0, corner_1, corner_2, corner_3 = [0, 0], [0, a_width], [a_high, 0], [a_high, a_width]
        corner_0_p = self.get_perspective_point(matrix, corner_0)
        corner_1_p = self.get_perspective_point(matrix, corner_1)
        corner_2_p = self.get_perspective_point(matrix, corner_2)
        corner_3_p = self.get_perspective_point(matrix, corner_3)

        background = b.copy()
        cv2.line(background, corner_0_p, corner_1_p, self.boundary_color, self.boundary_width)
        cv2.line(background, corner_1_p, corner_3_p, self.boundary_color, self.boundary_width)
        cv2.line(background, corner_3_p, corner_2_p, self.boundary_color, self.boundary_width)
        cv2.line(background, corner_2_p, corner_0_p, self.boundary_color, self.boundary_width)
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


class RANSAC:
    def __init__(self, max_iter, inner_ratio, threshold, kp1, des1, kp2, des2):  # 初始化
        self.max_iter = max_iter
        self.inner_ratio = inner_ratio
        self.threshold = threshold
        self.in_pt_num = 0
        self.kp1, self.des1, self.kp2, self.des2 = kp1, des1, kp2, des2
        self.kp1_index, self.kp2_index = range(len(kp1)), range(len(kp2))
        self.good_kp_threshold = 0.8

    def get_good_nearest_sift_kp(self, single_kp1_index):  # 获得高质量的尺度不变关键点
        """
        引入高质量尺度不变点判断，如最小方差较次小方差小 self.good_kp_threshold 倍，则图2中点为图1中点的高质量尺度不变关键点
        :return index 图2中关键点位置值， good_label标明是否为高质量关键点:
        """
        index = 0  # 关键点顺序值
        good_label = 0
        diff = np.add.reduce(np.square(self.des2 - self.des1[single_kp1_index]), axis=1)
        diff_sort = np.sort(diff)
        minimum, min_2nd = diff_sort[0], diff_sort[1]
        if minimum / min_2nd < self.good_kp_threshold:
            good_label = 1
            index = np.argsort(diff)[0]
        return [index, good_label]

    def get_conrespond_table(self):  # 获得图1中各点对应图2中点的映射表
        conrespond_table = []
        for _i in self.kp1_index:
            _index, _good_label = self.get_good_nearest_sift_kp(_i)
            if _good_label == 1:
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
        return np.sum(result), matrix, np.where(result == 1)

    def compare_refresh(self):  # 比较内点数量，刷新内点数量和单应性矩阵
        conrespond_table = np.array(self.get_conrespond_table())
        inner_count = 0
        matrix = []
        for _i in range(self.max_iter):
            new_inner_count, new_matrix, new_inner_point_index = self.count_inner_point(conrespond_table)
            if new_inner_count > inner_count:
                inner_count = new_inner_count
                matrix = new_matrix
            if 1.0 * inner_count / len(conrespond_table) > self.inner_ratio:
                break
        return inner_count, matrix


class ObjectTrack:  # 物体追踪
    def __init__(self, video_path, max_iteration, inner_ratio, threhold):
        self.VideoCapture = cv2.VideoCapture(video_path)  # 捕获视频
        self.fps = self.VideoCapture.get(cv2.CAP_PROP_FPS)  # 帧速
        self.fsize = (int(self.VideoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),  # 尺寸
                      int(self.VideoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fNUMS = self.VideoCapture.get(cv2.CAP_PROP_FRAME_COUNT)  # 帧数
        self.save_video_filename = './object_tracking_result.mp4'  # 追踪结果存储位置
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 写入格式
        self.max_iteration = max_iteration  # 每帧匹配时最大迭代数
        self.inner_ratio = inner_ratio  # 每帧匹配时内点率
        self.threshold = threhold  # 每帧匹配时内点判断阈值

    @staticmethod
    def draw_target_box_by_move_mouse(event, x, y, flags, para):  # 选择追踪目标
        global point1, point2
        ini_frame_copy = ini_frame.copy()
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            point1 = (x, y)
            cv2.circle(ini_frame_copy, point1, 10, (0, 255, 0), 5)
            cv2.imshow('image', ini_frame_copy)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
            cv2.rectangle(ini_frame_copy, point1, (x, y), (255, 0, 0), 5)
            cv2.imshow('image', ini_frame_copy)
        elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
            point2 = (x, y)
            cv2.rectangle(ini_frame_copy, point1, point2, (0, 0, 255), 5)
            cv2.imshow('image', ini_frame_copy)

    def get_tracking_target(self, frame):  # 返回框选目标
        global ini_frame
        ini_frame = frame.copy()
        # 创建图像与窗口并将窗口与回调函数绑定
        cv2.namedWindow('image')
        cv2.putText(ini_frame, 'Hold Left Button and Drag to Frame the Target',
                    (int(self.fsize[0] * 0.1), int(self.fsize[1] * 0.05)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
        cv2.putText(ini_frame, 'Press ESC to Exit.', (int(self.fsize[0] * 0.35), int(self.fsize[1] * 0.1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
        cv2.setMouseCallback('image', self.draw_target_box_by_move_mouse)
        while True:
            cv2.imshow('image', ini_frame)
            key = cv2.waitKey(0)
            if key == 27:  # 矩形
                break
        cv2.destroyAllWindows()
        min_x, min_y = min(point1[0], point2[0]), min(point1[1], point2[1])
        width, height = abs(point1[0] - point2[0]), abs(point1[1] - point2[1])
        return ini_frame[min_y:min_y + height, min_x:min_x + width]

    @staticmethod
    def tracking_one_frame(img_target_shape, img_frame, max_iteration, inner_ratio,
                           threshold, kp_target, des_target, kp_frame, des_frame):  # 在单一帧中匹配
        inner_count, matrix = RANSAC(max_iteration, inner_ratio, threshold, kp_target, des_target,
                                     kp_frame, des_frame).compare_refresh()
        return ShowImg.add_bounding_box(img_target_shape, img_frame, np.array(matrix))

    def tracking(self):  # 所有帧中匹配
        start = time.time()
        available, frame = self.VideoCapture.read()
        tracking_target = self.get_tracking_target(frame)
        tracking_target_shape = tracking_target.shape
        sift_img = SIFT()
        kp_target, des_target = sift_img.detect_compute(tracking_target)
        i = 0
        video_writer = cv2.VideoWriter(self.save_video_filename, self.fourcc, self.fps, self.fsize)
        while available:
            kp_frame, des_frame = sift_img.detect_compute(frame)
            frame_boxed = self.tracking_one_frame(tracking_target_shape, frame, self.max_iteration, self.inner_ratio,
                                                  self.threshold, kp_target, des_target, kp_frame, des_frame)
            if i % int(10) == 0:
                print(f"total_frame: {self.fNUMS}  current_frame: {i}  cost_time: {time.time() - start}")
                plt.imshow(ShowImg.bgr2rgb(frame_boxed))
                plt.show()
            video_writer.write(frame_boxed)
            available, frame = self.VideoCapture.read()  # 获取下一帧
            i = i + 1
        video_writer.release()
        self.VideoCapture.release()

    def play_video(self):  # 播放匹配结果
        video_capture = cv2.VideoCapture(self.save_video_filename)
        available, frame = video_capture.read()
        cv2.putText(frame, 'Push ESC to Quit', (int(self.fsize[0] / 2), int(self.fsize[1] / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (55, 255, 155), 2)
        while available:
            cv2.imshow('frame', frame)
            time.sleep(0.02)
            available, frame = video_capture.read()
            if cv2.waitKey(1) == 27:
                break
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ShowImg = ShowImg()  # 显示图片
    object_tracking = ObjectTrack('./object_tracking_video.mp4', 2000, 0.5, 1)
    object_tracking.tracking()
    object_tracking.play_video()
