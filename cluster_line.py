from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from calculate_cross import calculate_cross
import cv2

def dbscan(lines, polars, end_point, epss,MinPts, epss2, MinPts2):   # MinPts2建议为2
    '''
    :param feature0:
    :param MinPts:
    :return labels,unique_labels:
    '''
    if len(polars) == 0:
        # print('polars为空')
        return np.array([]), set()
    st_polars = StandardScaler().fit_transform(polars)
    db = DBSCAN(eps=epss, min_samples=MinPts).fit(st_polars)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print('总共聚类数：', n_clusters_)
    unique_labels = set(labels)
    try:
        unique_labels.remove(-1)
    except:  # KeyError
        pass
        # print('第一次聚类中没有-1类')
    else:
        pass

    # 二次聚类
    lines = lines.reshape([lines.shape[0], lines.shape[2]])
    for k in unique_labels:
        class_member_mask = (labels == k)
        sline = lines[class_member_mask] # 选择第k类的端点集合

        sline2line_map = np.empty(shape=[0,1], dtype=int)
        j=0
        while j<len(class_member_mask):
            if class_member_mask[j] == True:
                sline2line_map = np.append(sline2line_map, [j])
            j += 1


        midpoint = np.empty(shape=[0,2])
        for i in range(sline.shape[0]) : # 对于所有的直线
            x1, y1, x2, y2 = sline[i]
            mp = [(x1 + x2) / 2, (y1 + y2) / 2]
            midpoint = np.append(midpoint, [mp], axis=0)
        a = -1.218e-3
        b = 1.838
        midpoint[:, 1] = a * (midpoint[:, 1] - end_point[1]) ** 2 + b * (
        midpoint[:, 1] - end_point[1])  # 对中点集做变换，压缩离无穷远点较远的点，拉伸离无穷远点较近的点

        db2 = DBSCAN(eps=epss2, min_samples=MinPts2).fit(midpoint)
        labels2 = db2.labels_
        unique_labels2 = set(labels2)
        try:
            unique_labels2.remove(-1)
        except:  # KeyError
            # print('第二次聚类中第', k,'类没有-1类')
            pass
        else:
            pass
        number_unique_labels2 = []
        for k2 in unique_labels2:
            class_member_mask2 = (labels2 == k2)
            midpoint0 = midpoint[class_member_mask2]
            number_unique_labels2.append(len(midpoint0))
        if number_unique_labels2 != []:
            max_k2 = number_unique_labels2.index(max(number_unique_labels2)) # 数目最多的簇
            not_max = (labels2 != max_k2)
        else:   # 若为空，说明labels2里全是-1类，孤立点，则全部移除
            not_max = labels2
        line_not_max = sline2line_map[not_max]
        for i in line_not_max:
            labels[i] = -2    # 被移除的点记为-2类

        # colors = [plt.cm.Spectral(each)
    #           for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, colors):
    #     class_member_mask = (labels == k)
    #     polar = feature[class_member_mask]
    #     number_unique_labels.append(len(polar))
    #     plt.plot(polar[:, 0], polar[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=5)
    #     # xy = lines.reshape([lines.shape[0]*2,lines.shape[2]//2])[class_member_mask & ~core_samples_mask]
    #     # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)
    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.show()
    return labels,unique_labels

def slope_check(lines,slope_threshold, end_point, image_width, cc = 0.3):
    '''
    斜率检测，去除不符合实际斜率要求的直线
    :param lines:
    :param slope_threshold:
    :param end_point: 消失点
    :param cc: 参数，衡量与无穷远点的远近
    :return:
    '''
    new_lines = []
    left = max(end_point[0] - cc * image_width, 0)
    right = min(end_point[0] + cc * image_width, image_width)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        if x2 - x1 == 0.:
            slope = 999.
            end_point0 = x1
        else:
            slope = (y2-y1)/(x2-x1)
            if slope == 0: slope = 1e-15
            intercept = y1-slope*x1
            end_point0 = (end_point[1] -intercept)/slope
        if abs(slope) > slope_threshold and left < end_point0 < right:
            new_lines.append(line)

    return np.array(new_lines)

def line_change(lines):
    '''
    转换为极坐标系下坐标
    :param lines: 霍夫变换的线输出（点集）
    :return: 线集的极坐标下
    '''
    if len(lines) == 0: # 若lines为空，则返回空数组
        # print('lines为空')
        return lines
    lines = lines.reshape([lines.shape[0],lines.shape[2]])
    d = lines[:,1] - lines[:,3]
    d[d==0] = 1e-10
    theta = np.arctan(-(lines[:,0] - lines[:,2])/d)
    rho = lines[:,0] * np.cos(theta) + lines[:,1] * np.sin(theta)
    return np.vstack((theta,rho))


def select_lines(img, lines, polars, labels, unique_labels, end_point, cc, n_max ):
    '''
    :param img: 图片
    :param lines: 霍夫变换得到的线（2点组合）
    :param polars: 极坐标系下的线集（极坐标组合）
    :param labels: DBSCAN聚类算法得到的标签
    :param unique_labels: 独立标签
    :param n_max: 最多选择的车道数
    :param cc: 交叉点与消失点的距离系数
    :return lane_select:
    '''
    ####################################################################################################################
    # 计算每个类别对应的直线及其得分
    if unique_labels == set():
        lane_select_array = np.array([])
        # print('聚类数为0，不显示车道')
        return lane_select_array
    elif len(unique_labels) < n_max:
        # print('聚类数=',len(unique_labels),'< 所选车道数=',n_max,', 最大车道数n_max改为聚类数',len(unique_labels))
        n_max = len(unique_labels)
    lines = lines.reshape([lines.shape[0], lines.shape[2]])
    lane = [] # 选择的车道线
    lane_info = [] # 车道线信息
    for k in unique_labels:
        if k == -1: continue
        class_member_mask = (labels == k)
        spolar = polars[class_member_mask] # 选择第k类的极坐标
        sline = lines[class_member_mask] # 选择第k类的端点集合
        length = 0 # 总长度
        theta = 0 # 角度
        rho = 0 # 极坐标半径
        # black = np.zeros_like(img)
        for i in range(sline.shape[0]) : # 对于所有的直线
            x1, y1, x2, y2 = sline[i]  # line = [[x1, y1, x2, y2]]
            # cv2.line(black, (x1, y1), (x2, y2), color=[255, 255, 255], thickness=1)
            length0 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) # 对应这条边的长度
            length += length0 # 记录总长度
            theta0, rho0 = spolar[i]
            theta += length0 * theta0
            rho += length0 * rho0
        # cv2.imshow('b', black)
        # cv2.waitKey(0)

        theta = theta / length
        rho = rho / length

        x = np.vstack((sline[:, 0], sline[:, 2])).reshape(2 * sline.shape[0], 1)
        y = np.vstack((sline[:, 1], sline[:, 3])).reshape(2 * sline.shape[0], 1)

        if theta != 0:
            x1 = np.min(x)
            x2 = np.max(x)
            y1 = -np.cos(theta) / np.sin(theta) * x1 + rho / np.sin(theta)
            y2 = -np.cos(theta) / np.sin(theta) * x2 + rho / np.sin(theta)
            y3 = np.min(y)
            y4 = np.max(y)
            x3 = (rho - y3 * np.sin(theta)) / np.cos(theta)  # slope_check中已经排除了水平线的情况（theta=pi/2）
            x4 = (rho - y4 * np.sin(theta)) / np.cos(theta)
            xy = []
            if y3 <= y1 <= y4:
                xy.append(x1)
                xy.append(y1)
            if y3 <= y2 <= y4:
                xy.append(x2)
                xy.append(y2)
            if x1 <= x3 <= x2 and len(xy) <= 2:
                xy.append(x3)
                xy.append(y3)
            if x1 <= x4 <= x2 and len(xy) <= 2:
                xy.append(x4)
                xy.append(y4)
            if len(xy) != 4:
                # print('异常情况：加权直线与方框无交点   len(xy) != 4')
                xy = [x3, y3, x4, y4]
        else:
            xy = [rho, np.min(y), rho, np.max(y)]
        lane_length = np.sqrt((xy[0]-xy[2])**2+(xy[1] - xy[3])**2)
        lane.append(xy)
        lane_info.append([length, lane_length]) #
    lane_score = []
    c = 1
    for i in lane_info:
        lane_score.append(i[0] + c * i[1])

    ####################################################################################################################
    # 消除平行线和在消失点之前相交的线 + 超过3条车道后，新添加车道分数筛选
    lane_select = []
    lane_select_score = []
    count = 0
    while len(lane_select)<n_max and count<len(lane_score):
        ind = lane_score.index(max(lane_score))  # 得分最高的线的下标
        line0 = lane[ind]
        score0 = lane_score[ind]
        count += 1
        x0 = line0[0]
        y0 = line0[1]
        x1 = line0[2]
        y1 = line0[3]
        d = y1 - y0
        if d == 0:
            d = 1e-10
        theta1 = np.arctan(-(x1 - x0) / d)
        sign = 1
        for line in lane_select:
            x2 = line[0]
            y2 = line[1]
            x3 = line[2]
            y3 = line[3]
            d = y3-y2
            if d==0:
                d = 1e-10
            theta2 = np.arctan(-(x3 - x2) / d)
            two_lines = np.vstack((line0, line))
            cross = calculate_cross(two_lines)
            distence = np.sqrt((cross[0]-end_point[0])**2 + (cross[1]-end_point[1])**2)
            if distence > cc*(img.shape[0]-end_point[1]) or abs(theta1-theta2)<=10/180*np.pi :
                sign = 0
                # print('新添加车道和已有车道平行或在消失点前相交')
                break
        if sign == 1 and len(lane_select) >= 3 \
                and score0 < 500 and score0 / np.mean(lane_select_score) < 0.3 \
                and score0 / lane_select_score[-1] < 0.37:  # 新添加车道的分数不能太小：score0 < 500 and score0/np.mean(lane_select_score) < 0.3
                                                             # 也不能比前一条车道的分数小太多：score0/lane_select_score[-1] < 0.37
            sign = 0
            # print('超过3条车道后，新添加车道分数不符合条件')
        if sign == 1:
            lane_select.append(line0)
            lane_select_score.append(score0)
        lane_score[ind] = -score0  # 判断过的线段不再判断
    # if len(lane_select)<n_max:
        # print('select_lines条数为',len(lane_select),'< n_max=',n_max)
    lane_select_array = np.array(list(lane_select))
    return lane_select_array


def lane_check(two_lane):
    '''

    :param two_lane_points:
    :return:
    '''
    left_lane = []
    right_lane = []
    if two_lane[0][0] < two_lane[1][0]:
        left_lane = two_lane[0]
        right_lane = two_lane[1]
    else:
        left_lane = two_lane[1]
        right_lane = two_lane[0]



