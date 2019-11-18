import numpy as np
import cv2
import utils
import threshold
import matplotlib.pyplot as plt

def get_M_Minv():
    src = np.float32([[(203, 720), (585, 460), (695, 460), (1127, 720)]])
    dst = np.float32([[(320, 720), (320, 0), (960, 0), (960, 720)]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    return M,Minv


if __name__ == '__main__':
    file = 'E:\Code\lane_detect\\test_images\\test2.jpg'
    test1 = cv2.imread(file)
    thresholded = threshold.thresholding(test1)
    M, Minv = get_M_Minv()
    thresholded_wraped = cv2.warpPerspective(thresholded, M, test1.shape[1::-1], flags=cv2.INTER_LINEAR)
    plt.figure(figsize=(10,10))
    plt.subplot(2, 2, 1), plt.imshow(test1)
    plt.subplot(2, 2, 2), plt.imshow(thresholded, cmap='gray')
    plt.subplot(2,2,3),plt.imshow(thresholded_wraped, cmap='gray')
    plt.show()
