import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from moviepy.editor import VideoFileClip


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def get_slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


# thick red lines
def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    """workflow:
    1) examine each individual line returned by hough & determine if it's in left or right lane by its slope
    because we are working "upside down" with the array, the left lane will have a negative slope and right positive
    2) track extrema
    3) compute averages
    4) solve for b intercept
    5) use extrema to solve for points
    6) smooth frames and cache
    """
    global cache
    global first_frame
    y_global_min = img.shape[0]  # min will be the "highest" y value, or point down the road away from car
    y_max = img.shape[0]
    l_slope, r_slope = [], []
    l_lane, r_lane = [], []
    det_slope = 0.4
    α = 0.2
    # i got this alpha value off of the forums for the weighting between frames.
    # i understand what it does, but i dont understand where it comes from
    # much like some of the parameters in the hough function

    for line in lines:
        # 1
        for x1, y1, x2, y2 in line:
            slope = get_slope(x1, y1, x2, y2)
            if slope > det_slope:
                r_slope.append(slope)
                r_lane.append(line)
            elif slope < -det_slope:
                l_slope.append(slope)
                l_lane.append(line)
        # 2
        y_global_min = min(y1, y2, y_global_min)

    # to prevent errors in challenge video from dividing by zero
    if ((len(l_lane) == 0) or (len(r_lane) == 0)):
        print('no lane detected')
        return 1

    # 3
    l_slope_mean = np.mean(l_slope, axis=0)
    r_slope_mean = np.mean(r_slope, axis=0)
    l_mean = np.mean(np.array(l_lane), axis=0)
    r_mean = np.mean(np.array(r_lane), axis=0)

    if ((r_slope_mean == 0) or (l_slope_mean == 0)):
        print('dividing by zero')
        return 1

    # 4, y=mx+b -> b = y -mx
    l_b = l_mean[0][1] - (l_slope_mean * l_mean[0][0])
    r_b = r_mean[0][1] - (r_slope_mean * r_mean[0][0])

    # 5, using y-extrema (#2), b intercept (#4), and slope (#3) solve for x using y=mx+b
    # x = (y-b)/m
    # these 4 points are our two lines that we will pass to the draw function
    l_x1 = int((y_global_min - l_b) / l_slope_mean)
    l_x2 = int((y_max - l_b) / l_slope_mean)
    r_x1 = int((y_global_min - r_b) / r_slope_mean)
    r_x2 = int((y_max - r_b) / r_slope_mean)

    # 6
    if l_x1 > r_x1:
        l_x1 = int((l_x1 + r_x1) / 2)
        r_x1 = l_x1
        l_y1 = int((l_slope_mean * l_x1) + l_b)
        r_y1 = int((r_slope_mean * r_x1) + r_b)
        l_y2 = int((l_slope_mean * l_x2) + l_b)
        r_y2 = int((r_slope_mean * r_x2) + r_b)
    else:
        l_y1 = y_global_min
        l_y2 = y_max
        r_y1 = y_global_min
        r_y2 = y_max

    current_frame = np.array([l_x1, l_y1, l_x2, l_y2, r_x1, r_y1, r_x2, r_y2], dtype="float32")

    if first_frame == 1:
        next_frame = current_frame
        first_frame = 0
    else:
        prev_frame = cache
        next_frame = (1 - α) * prev_frame + α * current_frame

    cv2.line(img, (int(next_frame[0]), int(next_frame[1])), (int(next_frame[2]), int(next_frame[3])), color, thickness)
    cv2.line(img, (int(next_frame[4]), int(next_frame[5])), (int(next_frame[6]), int(next_frame[7])), color, thickness)

    cache = next_frame


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# image = mpimg.imread('whiteCarLaneSwitch.jpg')
# # printing out some stats and plotting the image
# print('This image is:', type(image), 'with dimensions:', image.shape)
# plt.imshow(image)
# plt.show()


def lane_img_pipeline(image):
    gray_image = grayscale(image)
    plt.imshow(gray_image, cmap='gray')  # cmap='gray' 才能显示灰度图像
    # 转换为HSV空间，利于提取yellow lane
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    plt.imshow(img_hsv)

    lower_yellow = np.array([20, 100, 100], dtype="uint8")
    upper_yellow = np.array([30, 255, 255], dtype="uint8")

    # cv2.inRange，<lower 和 >upper的像素点都置零，位于中间的置为255
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    plt.imshow(mask_yellow)

    mask_white = cv2.inRange(gray_image, 200, 255)
    plt.imshow(mask_white, cmap='gray')

    # 或操作，将yellow与white合并
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    plt.imshow(mask_yw, cmap='gray')

    # 与操作，将灰度图像淹膜处理，mask_yw以外的区域置零
    mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)
    plt.imshow(mask_yw_image, cmap='gray')

    # 高斯模糊（去噪）
    kernel_size = 5
    gauss_gray = gaussian_blur(mask_yw_image, kernel_size)
    plt.imshow(gauss_gray, cmap='gray')

    # Canny边缘检测
    low_threshold = 50
    high_threshold = 150
    canny_edges = canny(gauss_gray, low_threshold, high_threshold)
    plt.imshow(canny_edges, cmap='gray')

    # 划定ROI ，定义了一个上窄下宽梯形，顶点的坐标是笛卡尔坐标系下的
    imshape = image.shape
    lower_left = [imshape[1] / 9, imshape[0]]
    lower_right = [imshape[1] - imshape[1] / 9, imshape[0]]
    top_left = [imshape[1] / 2 - imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10]
    top_right = [imshape[1] / 2 + imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10]
    vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
    roi_image = region_of_interest(canny_edges, vertices)
    plt.imshow(roi_image, cmap='gray')

    # Hough Transform
    rho = 4
    theta = np.pi / 180
    # threshold is minimum number of intersections in a grid for candidate line to go to output
    threshold = 30
    min_line_len = 100
    max_line_gap = 180

    global first_frame
    line_image = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)
    plt.imshow(line_image, cmap='gray')
    result = weighted_img(line_image, image, α=0.8, β=1., λ=0.)
    plt.imshow(result)

    return result


first_frame = 1
white_output = 'test_video/HSV_process_solidWhiteRight_output.mp4'  # output文件名
clip1 = VideoFileClip('test_video/harder_challenge_video.mp4')  # 读入input video
print(clip1.fps)  # frames per second 25, 默认传给write
white_clip = clip1.fl_image(lane_img_pipeline)  # 对每一帧都执行lane_img_pipeline函数，函数返回的是操作后的image
white_clip.write_videofile(white_output, audio=False)
