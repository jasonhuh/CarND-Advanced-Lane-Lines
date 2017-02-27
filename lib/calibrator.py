#!/usr/bin/python

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


class CameraCalibrator:

    """
    This class is responsible for providing image calibration
    and image undistortion
    """
    def __init__(self):
        self.__mtx = None
        self.__dist = None
        self.__calibrated = False
        self.__objpoints = None
        self.__imgpoints = None

    def load_calibration_points(self, images_path, visualize=False):
        # prepare object points and image points from all the images
        """
        :rtype: object
        """
        if len(images_path) == 0:
            raise ValueError('No images path provided')

        images = glob.glob(images_path)

        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

                if visualize:
                    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.set_title('Original Image {}'.format(img.shape))
                    ax1.imshow(img)

                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                    ax2.set_title('Chessboard corners {}'.format(img.shape))
                    ax2.imshow(img)
            else:
                if visualize:
                    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    ax1.set_title('Original Image {}'.format(img.shape))
                    ax1.imshow(img)

                    ax2.set_title('Chessboard not found')

        self.__objpoints = objpoints
        self.__imgpoints = imgpoints

    def calibrate_camera(self, src_img):
        if self.__objpoints is None or self.__imgpoints is None:
            raise ValueError('Image points do not exist')

        img_size = (src_img.shape[1], src_img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.__objpoints, self.__imgpoints, img_size, None, None)
        self.__mtx = mtx
        self.__dist = dist
        self.__calibrated = True

    def undistort(self, src_img):
        if self.__calibrated == False:
            self.calibrate_camera(src_img)
        return cv2.undistort(src_img, self.__mtx, self.__dist, None, self.__mtx)