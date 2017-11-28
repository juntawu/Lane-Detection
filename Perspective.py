import numpy as np
import cv2

def flatten_perspective(image):
    (h, w) = (image.shape[0], image.shape[1])  # 获得图片的高度和宽度，h为图片的高度，w为图片的宽度
    source = np.float32([[400, 350], [900, 350], [0, 1024], [1280, 1024]])
    destination = np.float32([[0, 0], [1280, 0], [750, 1024], [850, 1024]])
    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    unwarp_matrix = cv2.getPerspectiveTransform(destination, source)
    return (cv2.warpPerspective(image, transform_matrix, (w, h)), unwarp_matrix)

def region_of_interest(img, vertices):
    '''
    提取感兴趣区域
    :param img:
    :param vertices:
    :return:
    '''
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask,vertices,ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image