import utils
import numpy as np
import matplotlib.pyplot as plt
import cv2

file = "E:\Code\lane_detect\\test_images\straight_lines1.jpg"
test_img1 = cv2.imread(file)

def thresholding(img):
    x_thresh = utils.abs_sobel_thresh(img, orient='x', thresh_min=10 ,thresh_max=230)
    mag_thresh = utils.mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 150))
    dir_thresh = utils.dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3))
    hls_thresh = utils.hls_select(img, thresh=(180, 255))
    lab_thresh = utils.lab_select(img, thresh=(155, 200))
    luv_thresh = utils.luv_select(img, thresh=(225, 255))
    #Thresholding combination
    threshholded = np.zeros_like(x_thresh)
    threshholded[((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1)) | (lab_thresh == 1) | (luv_thresh == 1)] = 1

    return threshholded

if __name__ == '__main__':

    binary_output = thresholding(test_img1)
    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1),plt.imshow(test_img1)
    plt.subplot(2,1,2), plt.imshow(binary_output)

    plt.show()