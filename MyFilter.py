import cv2
import numpy as np
from PIL import Image

def Color_Filter(image,white_threshold,yellow_threshold):
    '''
    颜色滤波，先滤除其中的黄色和白色部分出来
    :param img: 原图，我想先做HSV变换
    :param white_threshold: 白色线的阈值
    :param yellow_threshold: 黄色线的阈值
    :return:
    '''
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(img,white_threshold[0],white_threshold[1]) # 白色区域
    yellow_mask = cv2.inRange(img,yellow_threshold[0],yellow_threshold[1]) # 黄色区域
    white_image = cv2.bitwise_and(image,image,mask=white_mask)
    yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)
    image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)
    return image2 # 最终颜色滤波后的图像

def GuassianFilter(image,kernel_size):
    '''
    高斯滤波
    :param image:
    :param kernel_size:
    :return:
    '''
    return cv2.GaussianBlur(image, (kernel_size,kernel_size),0)

def CannyFilter(image, low_threshold, high_threshold):
    return cv2.Canny(image,low_threshold,high_threshold)


def GaborFilter(img, us, vs, kernel_size, sigma, gamma, ps):
    '''
    方向可调滤波器
    :param img: 图片,灰度图
    :param us: 尺度
    :param vs: 方向
    :param kernel_size: 滤波器核大小
    :param sigma: sigma 带宽，取常数5
    :param gamma: gamma 空间纵横比，一般取1
    :param ps: psi 相位，一般取0
    :return:
    '''
    # img_ = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_f = np.array(img, dtype=np.float32)
    img_f /= 255.
    kernel = cv2.getGaborKernel((kernel_size,kernel_size),sigma,vs*np.pi/180,us,gamma,ps)
    mask = np.power(cv2.filter2D(img_f,cv2.CV_32F,kernel),2)
    return mask


def BilateralFilter(img, d, sigmacolor, sigmaspace):
    '''
    滤除低频噪声，保持边缘信息
    双边滤波器：http://blog.csdn.net/on2way/article/details/46828567
    什么是双边滤波器：http://blog.csdn.net/abcjennifer/article/details/7616663
    :param img:
    :param d: 邻域直径
    :param sigmacolor: 空间相似性高斯函数标准差
    :param sigmaspace: 灰度值相似性高斯函数标准差
    :return:
    '''
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigmacolor, sigmaSpace=sigmaspace)

def EqualizeHist(img):
    '''
    直方图均衡化
    :param img:
    :return:
    '''
    return cv2.equalizeHist(img)

def gradient_abs_value_mask(image, sobel_kernel=3, axis='x', threshold=(0, 255)):
    """
    Masks the image based on gradient absolute value.
    制作一个梯度绝对值的mask，这里利用sobel算子进行边缘检测
    Parameters
    ----------
    image           : Image to mask.
    sobel_kernel    : Kernel of the Sobel gradient operation.
    axis            : Axis of the gradient, 'x' or 'y'.
    threshold       : Value threshold for it to make it to appear in the mask.

    Returns
    -------
    Image mask with 1s in activations and 0 in other pixels.
    """
    # Take the absolute value of derivative in x or y given orient = 'x' or 'y'
    if axis == 'x':
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if axis == 'y':
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel = np.uint8(255 * sobel / np.max(sobel))
    # Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    mask = np.zeros_like(sobel)
    # Return this mask as your binary_output image
    mask[(sobel >= threshold[0]) & (sobel <= threshold[1])] = 1
    return mask

def gradient_magnitude_mask(image, sobel_kernel=3, threshold=(0, 255)):
    """
    Masks the image based on gradient magnitude.
    制作一个基于梯度值大小的mask
    Parameters
    ----------
    image           : Image to mask.
    sobel_kernel    : Kernel of the Sobel gradient operation.
    threshold       : Magnitude threshold for it to make it to appear in the mask.

    Returns
    -------
    Image mask with 1s in activations and 0 in other pixels.
    """
    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    magnitude = (magnitude * 255 / np.max(magnitude)).astype(np.uint8)
    # Create a binary mask where mag thresholds are met
    mask = np.zeros_like(magnitude)
    mask[(magnitude >= threshold[0]) & (magnitude <= threshold[1])] = 1
    # Return this mask as your binary_output image
    return mask

def gradient_direction_mask(image, sobel_kernel=3, threshold=(0, np.pi / 2)):
    """
    Masks the image based on gradient direction.
    制作一个基于梯度方向的mask
    Parameters
    ----------
    image           : Image to mask.
    sobel_kernel    : Kernel of the Sobel gradient operation.
    threshold       : Direction threshold for it to make it to appear in the mask.

    Returns
    -------
    Image mask with 1s in activations and 0 in other pixels.
    """
    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients and calculate the direction of the gradient
    direction = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    # Create a binary mask where direction thresholds are met
    mask = np.zeros_like(direction)
    # Return this mask as your binary_output image
    mask[(direction >= threshold[0]) & (direction <= threshold[1])] = 1
    return mask

def color_threshold_mask(image, threshold=(0, 255)):
    """
    Masks the image based on color intensity.
    利用颜色信息滤除相应的无关信息，这里返回一个用于颜色滤除的mask
    Parameters
    ----------
    image           : Image to mask.
    threshold       : Color intensity threshold.

    Returns
    -------
    Image mask with 1s in activations and 0 in other pixels. 对颜色进行提取
    """
    mask = np.zeros_like(image)
    mask[(image > threshold[0]) & (image <= threshold[1])] = 255
    return mask

def sobel_color_filter(image, separate_channels=False):
    """
    Masks the image based on a composition of edge detectors: gradient value,
    gradient magnitude, gradient direction and color.

    Parameters
    ----------
    image               : Image to mask.
    separate_channels   : Flag indicating if we need to put masks in different color channels.

    Returns
    -------
    Image mask with 1s in activations and 0 in other pixels.
    """
    # Convert to HLS color space and separate required channel 图片先做颜色转换并化为float类型
    hls = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:, :, 2]
    # Get a combination of all gradient thresholding masks
    gradient_x = gradient_abs_value_mask(s_channel, axis='x', sobel_kernel=3, threshold=(20, 100))
    gradient_y = gradient_abs_value_mask(s_channel, axis='y', sobel_kernel=3, threshold=(20, 100))
    magnitude = gradient_magnitude_mask(s_channel, sobel_kernel=3, threshold=(20, 100))
    direction = gradient_direction_mask(s_channel, sobel_kernel=3, threshold=(0.7, 1.3))
    gradient_mask = np.zeros_like(s_channel)
    gradient_mask[((gradient_x == 1) & (gradient_y == 1)) | ((magnitude == 1) & (direction == 1))] = 1
    # Get a color thresholding mask
    color_mask = color_threshold_mask(s_channel, threshold=(100, 255))

    if separate_channels:
        return np.dstack((np.zeros_like(s_channel), gradient_mask, color_mask))
    else:
        mask = np.zeros_like(gradient_mask)
        mask[(gradient_mask == 1) | (color_mask == 1)] = 1
        return mask # 这个是最终的mask，基于颜色和梯度信息的mask