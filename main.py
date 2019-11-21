import numpy as np
import matplotlib.pyplot as plt
import cv2
import camera

checher_path = 'camera_cal'
image_path = 'test.images'
out_img = 'out_img'



def main():

    #相机校正
    ca