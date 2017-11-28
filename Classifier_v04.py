import numpy as np
import cv2
from HoughLine import HoughLineTransform
import matplotlib.pyplot as plt

################### 待调参数 ############################
# 在输入的直线附近取像素的区域大小
region_size = 20

# 白色判断的 bgr 阈值
# white_thres = 170
# hsv 颜色阈值空间，需与main.py 颜色阈值参数一致
# white_low_thres = [0, 0, 150]
# white_high_thres = [190, 50, 255]
# yellow_low_thres = [15, 50, 50]
# yellow_high_thres = [40, 200, 255]
#多数场景比较通用
white_low_thres = [0, 0, 200]
white_high_thres = [100, 50, 255]
yellow_low_thres = [15, 90, 50]
yellow_high_thres = [40, 200, 255]

# 车道线虚实判断的长度阈值与间断数阈值
length_thres = 300
gap_thres = 2
length_ratio_up = 3.5
length_ratio_down = 0.8
##########################################################



# 判断车道线上的点像素值是否为白色
def isWhite(pixel):
    # hsv空间判断
    if( (white_low_thres[0] <= pixel[0] <= white_high_thres[0]) and (white_low_thres[1] <= pixel[1] <= white_high_thres[1])
                and (white_low_thres[2] <= pixel[2] <= white_high_thres[2]) ):
    # bgr空间判断
    # if ((white_thres <= pixel[0] <= 255) and (white_thres <= pixel[1] <= 255) and (white_thres <= pixel[2] <= 255)):
    # if ((white_thres <= pixel[0] <= 255) ):
        color_flag = True
    elif( (yellow_low_thres[0] <= pixel[0] <= yellow_high_thres[0]) and (yellow_low_thres[1] <= pixel[1] <= yellow_high_thres[1])
                and (yellow_low_thres[2] <= pixel[2] <= yellow_high_thres[2]) ):
        color_flag = False
    else:
        color_flag = True
    return color_flag


'''
通过车道线像素点的纵坐标判断虚实
鲁棒性不强，对于趋于水平又出现多个gap的车道线，会出现误检
后续可考虑用欧氏距离代替纵坐标
'''
# 判断车道线是否为虚线
def isDashed(gap_num, unbroken_line):
    solid = [i for i in range(len(unbroken_line)) if unbroken_line[i] >= length_thres]
    solid_num = len(solid)
    if (solid_num >= 1):
        type_flag = False
    else:
        if (gap_num >= gap_thres):
            type_flag = True
        else:
            type_flag = False
    return type_flag


def Classifier(image, color_edges, type_edges, src_bottom_point, src_top_point, lane_index, lane_num):
    '''
    image: 原图像
    color_edges: 用于颜色检测的边缘图
    type_edges: 用于虚实检测的边缘图
    src_bottom_point: 车道线下端点
    src_top_point: 车道线上端点
    '''
    try:
        # 转换颜色空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # plt.imshow(hsv_image)
        # plt.show()
        # 用来标示直线位置的列表
        lane_position = []

        # 根据聚类直线的端点坐标，在原图像中将直线标示出来

        # 车道线的掩模
        lane_mask = np.zeros_like(color_edges)
        cv2.line(lane_mask, src_bottom_point, src_top_point, [255, 255, 255], region_size)
        ret, thresh = cv2.threshold(lane_mask, 127, 255, cv2.THRESH_BINARY)
        # print(color_edges.shape)
        corlor_lane_edges = cv2.bitwise_and(color_edges, thresh)
        type_lane_edges = cv2.bitwise_and(type_edges, thresh)
        # print(lane_edges.shape)
        # cv2.imshow('lane_mask',lane_mask)
        # cv2.waitKey(0)

        ############################## 检测颜色 ##########################################
        # 在边缘图上找像素点
        index = corlor_lane_edges[:, :] > 127
        # hsv空间像素值
        lane_points =  hsv_image[index]
        # bgr空间像素值
        # lane_points = image[index]
        # 二维列表对列求和，之后求均值
        # sum_points = [sum(lane_points[:,0]), sum(lane_points[:,1]), sum(lane_points[:,2])]
        sum_points = map(sum, zip(*lane_points))
        average_points = [element / len(lane_points) for element in sum_points]
        # print(average_points)
        color_flag = isWhite(average_points)
        # if (color_flag == True):
        #     print("白色")
        # else:
        #     print("黄色")
        ##################################################################################

        ###################################### 检测虚实 #############################################
        index_mat = np.mat(type_lane_edges[:, :] > 127)
        index_y = []
        for i in range(index_mat.shape[0]):
            if (True in index_mat[i, :]):
                index_y.append(i)
        # index_y 的元素前后做差
        dif_y = [y2 - y1 for (y2, y1) in zip(index_y[1:], index_y[:-1])]
        # 寻找index_y 中元素值大于2的下标,即寻找gap的位置
        gap_index = [j for j in range(len(dif_y)) if dif_y[j] >= 3]
        # 统计index_y 的元素前后差值大于2 的个数，即线段的间隔数
        gap_num = len(gap_index)
        # 计算gap值连续为1 的长度，即实线长度
        gap_index.append(len(dif_y) - 1)
        gap_index.insert(0, 0)
        unbroken_line = [g2 - g1 for (g2, g1) in zip(gap_index[1:], gap_index[:-1])]
        # 对离中心比较远的线的长度做下矫正
        if(1==lane_index or lane_index == lane_num):
            unbroken_line = [unbroken_line[i] * length_ratio_up for i in range(len(unbroken_line))]
        else:
            unbroken_line = [unbroken_line[i] * length_ratio_down for i in range(len(unbroken_line))]
        # print(unbroken_line, gap_num)
        type_flag = isDashed(gap_num, unbroken_line)
        # if (type_flag == True):
        #     print("虚线")
        # else:
        #     print("实线")
        # cv2.imshow('lane_edges', lane_edges)
        # cv2.waitKey(0)
        ################################################################################################

        lines = HoughLineTransform(type_lane_edges, 'p')
        # line_img = np.zeros((*img.shape, 3), dtype=np.uint8)  # 3-channel RGB image
        shape = image.shape
        line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        line_img1 = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        # Find slopes of all lines
        # But only care about lines where abs(slope) > slope_threshold 只关心斜率大于slope_threshold的直线
        slope_threshold = 0.2
        line_num = 0
        new_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]
            line_num += 1

            # Calculate slope
            if x2 - x1 == 0.:  # corner case, avoiding division by 0
                slope = 999.  # practically infinite slope
            else:
                slope = (y2 - y1) / (x2 - x1)


            # Filter lines based on slope 排除掉斜率明显不正常的直线
            if abs(slope) > slope_threshold:
                new_lines.append(line)

        for line in new_lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_img1,(x1,y1),(x2,y2),[0,0,255],1)
        # cv2.imshow('line', line_img1)
        # cv2.waitKey(0)

        # 对霍夫检测的线的端点进行直线拟合
        points = np.array(new_lines).reshape(np.array(new_lines).shape[0] * 2, 2)
        rows, cols = image.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        line_m = vy / vx
        line_b = y - x * line_m
        bottom_y = int(rows)
        bottom_x = int((bottom_y - line_b) / line_m)
        top_y = 0
        top_x = int((top_y - line_b) / line_m)
        # cv2.line(image, (bottom_x, bottom_y), (top_x, top_y), [0, 0, 255], 2)

        left_x = 0
        left_y = int(left_x * line_m + line_b)
        right_x = cols
        right_y = int(right_x * line_m + line_b)

        # 在检测到的车道线上取10个点
        points_num = 20
        height_per = 0.3
        if (bottom_x < 0):
            density_y = int((left_y - rows * height_per) / points_num)
            start_y = left_y
        elif (bottom_x > cols):
            density_y = int((right_y - rows * height_per) / points_num)
            start_y = right_y
        else:
            density_y = int((bottom_y - rows * height_per) / points_num)
            start_y = bottom_y
        for i in range(points_num):
            temp_y = start_y - i * density_y
            temp_x = (temp_y - line_b) / line_m
            lane_position.append([int(temp_x), int(temp_y)])
        # cv2.imshow('annotated_image', image)
        # cv2.waitKey(1)

    except:
        # lane_position = [ [1200, 1024], [1170, 989], [1140, 954], [1111, 919], [1081, 884], [1051, 849], [1022, 814], [992, 779], [962, 744], [932, 709], [903, 674], [873, 639], [843, 604], [814, 569], [784, 534], [754, 499], [725, 464], [695, 429], [665, 394], [635, 359] ]
        lane_position = [[1200, 1024], [1170, 989], [1140, 954]]
        color_flag = 1
        type_flag = 0
        print(lane_position)
        return lane_position, color_flag, type_flag

    # 返回车道线的类型及标识点
    return lane_position, color_flag, type_flag

