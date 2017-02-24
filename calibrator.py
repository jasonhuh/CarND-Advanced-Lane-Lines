import cv2
import numpy as np
import matplotlib.pyplot as plt


class CameraCalibrator():

    """
    This class is responsible for providing image calibration
    and image undistortion
    """
    def __init__(self, images):
        if len(images) == 0:
            raise ValueError('No images provided')
        self.__mtx = None
        self.__dist = None
        self.__calibrated = False

        self.__objpoints, self.__imgpoints = self.__get_calibration_points(images)

    def __get_calibration_points(self, images):
        # prepare object points and image points from all the images
        """
        :rtype: object
        """
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

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                plt.figure()
                plt.imshow(img)
                plt.show()
            else:
                print('not found')

        return objpoints, imgpoints

    def calibrate_camera(self, src_img):
        img_size = (src_img.shape[1], src_img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.__objpoints, self.__imgpoints, img_size, None, None)
        self.__mtx = mtx
        self.__dist = dist
        self.__calibrated = True

    def undistort(self, src_img):
        if self.__calibrated == False:
            self.calibrate_camera(src_img)
        return cv2.undistort(src_img, self.__mtx, self.__dist, None, self.__mtx)