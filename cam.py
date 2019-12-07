import cv2
import numpy as np
import utils
import matplotlib.pyplot as plt

def get_obj_img_points(images, grid=(9,6)):
    object_points = []
    img_points = []
    for img in images:
        object_point = np.zeros( (grid[0]*grid[1],3),np.float32 )
        object_point[:,:2]= np.mgrid[0:grid[0],0:grid[1]].T.reshape(-1,2)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, grid, None)
        if ret:
            object_points.append(object_point)
            img_points.append(corners)

    return object_points, img_points

def cal_undistort(img, objpoints, imgpoints):
    ret, mtx, dist,rvecs, tves = cv2.calibrateCamera(objpoints,imgpoints, img.shape[1::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst


def cam_calibration(chechers_imgs, lane_imgs):



    object_points, img_points = get_obj_img_points(chechers_imgs)


    undistorted = []
    for img in lane_imgs:
        img = cal_undistort(img, object_points, img_points)
        undistorted.append(img)
    return lane_imgs, undistorted



def display(in_imgs, out_imgs):
    plt.figure(figsize=(10,10))
    i = 0
    j = 1
    for img1, img2 in zip(in_imgs, out_imgs):
        plt.subplot(8, 2, j), plt.imshow(img1)
        j = j + 1
        plt.subplot(8, 2, j), plt.imshow(img2)
        j = j + 1
        i = i + 1
    plt.show()



if __name__ == '__main__':
    checker_path = 'camera_cal'
    imgs_path = 'test_images'
    checker_imgs = utils.get_images_by_dir(checker_path)
    lane_imgs = utils.get_images_by_dir(imgs_path)
    imgs, undistort = cam_calibration(checker_imgs, lane_imgs)
    display(imgs, undistort)