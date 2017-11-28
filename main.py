import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import Perspective as Perspective
import MyFilter as Filter
import cluster_line as Cluster
import HoughLine as HoughLine
import find_endpoint as fe
from XML_Creater import XML
from Classifier_v04 import Classifier
import Classifier_v04
import time


################################################## 待调参数 #######################################################
# 数据集路径设置
# path = 'G:\JuntaWu\lane_detection\MyLane_20171103\image\dataSet'
path= 'G:\JuntaWu\lane_detection\MyLane_20171103\image\\test'
# 生成的 xml 文件保存路径
xml_path = 'G:\JuntaWu\lane_detection\MyLane_20171103\MyLaneTrack_v08.7\TSD-Lane-Result'

# canny 滤波参数
low_threshold = 50
high_threshold = 80

# hsv 颜色阈值空间，需与Classifier.py 颜色阈值参数一致
white_low_thres = Classifier_v04.white_low_thres
white_high_thres = Classifier_v04.white_high_thres
yellow_low_thres = Classifier_v04.yellow_low_thres
yellow_high_thres = Classifier_v04.yellow_high_thres

# 检测车道线数
lane_num = 4
###################################################################################################################

folders = os.listdir(path)
for folder in folders:
    # 提示信息与计时
    print('Processing ' + folder + ' :')
    start_time = time.time() # 计时

    image_path = os.path.join(path, folder)
    image_files = os.listdir(image_path)
    # 创建xml文件，存储车道线信息
    # xml_creater = XML('TSD-Lane-%05d-Result.xml' % data_index )
    xml_creater = XML(folder + '-Result.xml')
    # 当前图片帧号
    frameIndex = 0
    # 存储上一帧检测结果，初始化
    previous_extend_lane = np.array([[1194, 1024,  574,  295],  [1194, 900,  574,  250] ])
    for file in image_files:
        try:
            img = cv2.imread(image_path + '\\' + file)
            image = img.copy()
            #######################################################################################################################
            # 预处理：滤波+边缘检测
            median = cv2.medianBlur(img, 5)

            img_1 = (Filter.GaborFilter(median, 5, 0, 3, 5, 1, 0)).astype(np.uint8)  # 利用gabor滤波器提取方向边缘信息
            img_2 = (Filter.GaborFilter(median, 5, 25, 3, 5, 1, 0)).astype(np.uint8)
            img_3 = (Filter.GaborFilter(median, 5, 50, 3, 5, 1, 0)).astype(np.uint8)
            img_4 = (Filter.GaborFilter(median, 5, 90, 3, 5, 1, 0)).astype(np.uint8)
            gabor = np.abs((img_1 + +img_2 + img_3 + img_4)).astype(np.uint8)

            gabor_edges = Filter.CannyFilter(gabor, low_threshold, high_threshold)
            end_point = fe.find_endpoint(img, gabor_edges)
            # cv2.imshow("gabor_result", gabor_edges)

            ROI = np.array([[(0,end_point[1]),(img.shape[1],end_point[1]),(img.shape[1],img.shape[0]),(0,img.shape[0])]],dtype=np.int32)
            roi = Perspective.region_of_interest(gabor_edges, ROI)
            #########################################################################################################################

            ###############################################################################################
            # 获得颜色、虚实检测所需的canny_edges
            # 先进行一轮颜色滤波
            img_color = Filter.Color_Filter(image, (np.array(white_low_thres), np.array(white_high_thres)),
                                            (np.array(yellow_low_thres), np.array(yellow_high_thres)))
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            _, img_thres = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
            # cv2.imshow("thres", img_thres)
            ###############################################################################################

            ################################################################################################################################
            # 检测当前帧车道线，检测不到时异常处理
            # 车道检测：直线检测+斜率筛选+聚类+车道挑选
            hough_lines = HoughLine.HoughLineTransform(roi, 'p')
            lines = Cluster.slope_check(hough_lines, 0.2, end_point, img.shape[1], 0.2)
            polars = (Cluster.line_change(lines)).T
            labels, unique_labels = Cluster.dbscan(lines, polars, end_point, 0.1, 3, 280,2)  # epss=0.1, MinPts=3, epss2=350, MinPts2=2)
            lane = Cluster.select_lines(img, lines, polars, labels, unique_labels, end_point, cc=0.2, n_max = lane_num)

            hough = np.zeros_like(img)
            # for line in hough_lines:
            #     cv2.line(hough, (int(line[0, 0]), int(line[0, 1])), (int(line[0, 2]), int(line[0, 3])), [0, 0, 255], 1)
            # cv2.imshow('hough', hough)

            slope_check = np.zeros_like(img)
            # for line in lines:
            #     cv2.line(slope_check, (int(line[0, 0]), int(line[0, 1])), (int(line[0, 2]), int(line[0, 3])), [0, 0, 255], 1)
            # cv2.imshow('slope_check', slope_check)

            ########################################### 显示检测直线的效果 ##############################################
            colors2 = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 0], [255, 255, 0], [255, 0, 255], [0, 255, 255],
                       [255, 255, 255]]  # 红、绿、蓝、黑、青、粉、黄、白   直线颜色按得分高到低从前往后排列
            y1 = img.shape[0]
            y2 = end_point[1]  # 取消失点下方的点
            extend_lane = []     # 延长后的车道
            for line, col in zip(lane, colors2):
                if line[0] == line[2]:
                    x1 = line[0]
                    x2 = line[0]
                else:
                    line_m = (line[1] - line[3]) / (line[0] - line[2])
                    line_b = line[3] - line_m * line[2]
                    x1 = int((y1 - line_b) / line_m)
                    x2 = int((y2 - line_b) / line_m)
                # cv2.line(img, (x1, y1), (x2, y2), col, 10)
                extend_lane.append([x1, y1, x2, y2])
            extend_lane = np.array(extend_lane)
            # cv2.imshow('img', img)
            #############################################################################################################

            # 更新上一帧检测到的车道线结果，以备当前帧检测不到车道线情况
            previous_extend_lane = extend_lane

            # 根据 x1 的数值大小，对检测到的车道线从左到右排序
            # 存储从左到右排好序的车道线端点
            lane_points= np.empty( shape=[0, extend_lane.shape[1] ] )  # lane_points：按x1从小到大排序的车道线
            iter_num = extend_lane.shape[0]
            for index in range( iter_num ):
                min_index = extend_lane[:, 0].argmin()
                lane_points = np.vstack( (lane_points, extend_lane[ min_index, :]) )
                extend_lane = np.delete(extend_lane, min_index, axis=0)
            # print(lane_points)

            # 对检测到的每条车道线进行颜色,类型判断,并将结果写入xml文件
            targetNumber = lane_points.shape[0]
            xml_creater.create_targetNumber(frameIndex, targetNumber)
            targetIndex = 0  # 目标车道线
            for i in range(targetNumber):
                # 检测不到时，异常如何处理 ？？？
                lane_position, color, type = Classifier(image, img_thres, img_thres, ( int(lane_points[i, 0]), int(lane_points[i, 1])), \
                                                        ( int(lane_points[i, 2]), int(lane_points[i, 3]) ), i+1, targetNumber )
                xml_creater.create_target(frameIndex, targetIndex, color, type, lane_position)
                targetIndex += 1

                ############################## 显示检测虚实、黄白的效果 ########################################
                # for point in lane_position:
                #     cv2.circle(image,(point[0], point[1] ), 3, [0, 0, 255], -1)
                if color == True:
                    if type == True:
                        col = [100,100,100] # 白虚线  灰色
                    else:
                        col = [0,0,0] # 白实线  黑色
                else:
                    if type == True:
                        col = [255,255,0] # 黄虚线 青色
                    else:
                        col = [0,255,0] # 黄实线  绿色
                cv2.line(image, ( int(lane_points[i, 0]), int(lane_points[i, 1])), \
                                                        ( int(lane_points[i, 2]), int(lane_points[i, 3]) ), col,7)
                #####################################################################################################
        except:
            pass
            # extend_lane = previous_extend_lane
            # # if(extend_lane)
            # lane_points = np.empty(shape=[0, extend_lane.shape[1]])  # lane_points：按x1从小到大排序的车道线
            # iter_num = extend_lane.shape[0]
            # for index in range(iter_num):
            #     min_index = extend_lane[:, 0].argmin()
            #     lane_points = np.vstack((lane_points, extend_lane[min_index, :]))
            #     extend_lane = np.delete(extend_lane, min_index, axis=0)
            # # print(lane_points)
            #
            # # 对检测到的每条车道线进行颜色,类型判断,并将结果写入xml文件
            # targetNumber = lane_points.shape[0]
            # xml_creater.create_targetNumber(frameIndex, targetNumber)
            # targetIndex = 0  # 目标车道线
            # for i in range(targetNumber):
            #     # 检测不到时，异常如何处理 ？？？
            #     lane_position, color, type = Classifier(image, img_thres, img_thres,
            #                                             (int(lane_points[i, 0]), int(lane_points[i, 1])), \
            #                                             (int(lane_points[i, 2]), int(lane_points[i, 3])), i + 1,
            #                                             targetNumber)
            #     xml_creater.create_target(frameIndex, targetIndex, color, type, lane_position)
            #     targetIndex += 1
            #
            #     ############################## 显示检测虚实、黄白的效果 ########################################
            #     # for point in lane_position:
            #     #     cv2.circle(image,(point[0], point[1] ), 3, [0, 0, 255], -1)
            #     if color == True:
            #         if type == True:
            #             col = [100, 100, 100]  # 白虚线  灰色
            #         else:
            #             col = [0, 0, 0]  # 白实线  黑色
            #     else:
            #         if type == True:
            #             col = [255, 255, 0]  # 黄虚线 青色
            #         else:
            #             col = [0, 255, 0]  # 黄实线  绿色
            #     cv2.line(image, (int(lane_points[i, 0]), int(lane_points[i, 1])), \
            #              (int(lane_points[i, 2]), int(lane_points[i, 3])), col, 7)
            #     #####################################################################################################
        ##############################################################################################################################
        frameIndex += 1

        # 显示检测效果
        # cv2.imshow("annotated_image", image)
        # cv2.waitKey(1)

    # 关闭xml文件
    xml_creater.close()


    ########################## 修改XML格式 #############################
    xml_new = open(xml_path + '\\' + folder + '-Result.xml', 'w', encoding='gb2312')
    with open( folder + '-Result.xml', 'r', encoding='utf-8') as f:
        xml_lines = f.readlines()
        count = 0
        for xml_line in xml_lines:
            if count == 0:
                count += 1
                xml_new.write(xml_line)
                continue
            if '.' in xml_line:
                new_line = ''
                fragments = xml_line.split('.')
                for fragment in fragments:
                    new_line += fragment
                xml_new.write(new_line)
                continue
            count += 1
            xml_new.write(xml_line)
    ########################################################################

    # 计时
    end_time = time.time()
    print(end_time - start_time)