import cluster_line as Cluster
import Perspective as Perspective
import HoughLine as HoughLine
from calculate_cross import calculate_cross
import numpy as np
import cv2


def find_endpoint(img, canny):

    end_point0 = [725, 331]  # 粗略消失点 [700,310]
    try: # select_lines之后的lane可能少于两条线
        ROI = np.array([[(0, end_point0[1]), (img.shape[1], end_point0[1]), (img.shape[1], img.shape[0]), (0, img.shape[0])]],
                       dtype=np.int32)
        roi = Perspective.region_of_interest(canny, ROI)
        #########################################################################################################################
        # 车道检测：直线检测+斜率筛选+聚类+车道挑选
        hough_lines = HoughLine.HoughLineTransform(roi, 'p')

        c = 0.3
        lines = Cluster.slope_check(hough_lines, 0.2, end_point0, img.shape[1], c)

        polars = (Cluster.line_change(lines)).T
        labels, unique_labels = Cluster.dbscan(lines, polars, end_point0, 0.1, 3, 280, 2)  # epss=0.1, MinPts=3, epss2=350, MinPts2=2)

        cc = 0.4
        lane = Cluster.select_lines(img, lines, polars, labels, unique_labels, end_point0, cc, n_max=2)

        end_point = calculate_cross(lane)
        if 0.25*img.shape[1] < end_point[0] < 0.75*img.shape[1] and 0.17*img.shape[0] < end_point[1] < 0.6*img.shape[0]:
        ######################## 画消失点 ###################################################################
        #     cv2.circle(img, tuple(end_point0), 10, [255, 0, 0], -1)
        #     cv2.circle(img, tuple(end_point), 10, [0, 255, 0], -1)
        #
        #     for line in lane:
        #         cv2.line(img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), [0, 255, 0], 10)
        #     # cv2.imshow('find_endpoing', img)
        #####################################################################################################
            return end_point
        else:
            print('消失点超出默认范围')
            return end_point0
    except:
        print('lane可能少于两条线')
        return end_point0
