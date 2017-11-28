import cv2
import numpy as np


rho = 2 # distance resolution in pixels of the Hough grid
theta = 1 * np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15	 # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10 # minimum number of pixels making up a line
max_line_gap = 20	# maximum gap in pixels between connectable line segments

def HoughLineTransform(image,type):
    '''
    进行霍夫变换
    :param image: 图片
    :param type: 进行何种霍夫变换，原始霍夫变换还是概率霍夫变换
    :return:
    '''
    if type == 'p':
        hough = cv2.HoughLinesP(image,rho,theta,threshold,np.array([]),minLineLength=min_line_length,maxLineGap=max_line_gap)
        if hough is None:
            return np.array([])
        return hough
    else:
        hough = cv2.HoughLines(image, rho, theta, threshold, np.array([]))
        if hough is None:
            return np.array([])
        return hough