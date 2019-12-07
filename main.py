import numpy as np
import matplotlib.pyplot as plt
import cv2
import cam
import threshold
import perspective_transform
import find_line1
import final_process
import utils

checher_path = 'camera_cal'
image_path = 'test.images'
out_img = 'out_img'



def main():

    binary_outputs = []
    thresholded_wrapeds = []
    results = []
    checker_path = 'camera_cal'
    imgs_path = 'test_images'
    checker_imgs = utils.get_images_by_dir(checker_path)
    lane_imgs = utils.get_images_by_dir(imgs_path)
    # 相机校正
    imgs, undistorts = cam.cam_calibration(checker_imgs, lane_imgs)
    #阈值过滤
    for undistort in undistorts:
        binary_output = threshold.thresholding(undistort)
        binary_outputs.append(binary_output)
    #透视变换
        M, Minv = perspective_transform.get_M_Minv()

        thresholded_wraped = cv2.warpPerspective(binary_output, M, undistort.shape[1::-1], flags=cv2.INTER_LINEAR)

    #检测车道边界、滑动窗口拟合
        left_fit, right_fit, left_lane_inds, right_lane_inds = find_line1.find_line(thresholded_wraped)
    #计算车道曲率、及车辆相对车道中心位置、显示信息
        curvature, distance_from_center = final_process.calculate_curv_and_pos(thresholded_wraped, left_fit, right_fit)
        result = final_process.draw_area(undistort, thresholded_wraped, Minv, left_fit, right_fit)
        final_process.draw_values(result, curvature, distance_from_center)
        results.append(result)



    cam.display(lane_imgs, results)

if __name__ == '__main__':
    main()


