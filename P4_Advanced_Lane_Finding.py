# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 19:12:46 2016

@author: macuser
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

PICKLE_FILE ='output/p4_pickle.p'

def calibrate_camera():
    objp = np.zeros((6*9, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    dist_pickle = pickle.load( open( PICKLE_FILE, "rb" ) )
    if dist_pickle is None:
    
        objpoints = []
        imgpoints = []
        
        images = glob.glob('camera_cal/calibration*.jpg')
        
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                
                cv2.drawChessboardCorners(img, (9, 6), corners, ret)
    #            cv2.imshow('img', img)
    #            cv2.waitKey(500)
            else:
                print('not found')
    
    #    cv2.destroyAllWindows()
    
        dist_pickle = {}
        dist_pickle["objpoints"] = objpoints
        dist_pickle["imgpoints"] = imgpoints
        
        pickle.dump(dist_pickle, open(PICKLE_FILE, 'wb'))
    
    else:
        print('Found pickle')
        objpoints = dist_pickle["objpoints"]
        imgpoints = dist_pickle["imgpoints"]    
    
    
    img = cv2.imread('test_images/test1.jpg')
    img_size = (img.shape[1], img.shape[0])
    print(img.shape)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    
    dst = cv2.undistort(img, mtx, dist, None, mtx)
#    cv2.imwrite('output/test_undist.jpg', dst)
    
#    dist_pickle["mtx"] = mtx
#    dist_pickle["dist"] = dist
    
#    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
#    ax1.imshow(img)
    #ax1.set_title('Original Image', fontsize=30)
#    ax2.imshow(dst)
#    ax2.set_title('Undistorted Image', fontsize=30)
    test_image(dst)

def test_image(frame_img):
#    src = np.zeros((4, 2), dtype = "float32")
#	dst = np.array([
#		[0, 0],
#		[maxWidth - 1, 0],
#		[maxWidth - 1, maxHeight - 1],
#		[0, maxHeight - 1]], dtype = "float32")    
#    corners = []
    img_size = (frame_img.shape[1], frame_img.shape[0])
#    src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
#    # For destination points, I'm arbitrarily choosing some points to be
#    # a nice fit for displaying our warped result 
#    # again, not exact, but close enough for our purposes
#    dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
#                                 [img_size[0]-offset, img_size[1]-offset], 
#                                 [offset, img_size[1]-offset]])

    offset = 150
    
    src = np.float32([[585,460],[730,460],[1055,750], [320,750]])

    line_image = np.copy(frame_img)*0
    cv2.line(line_image,(src[0][0],src[0][1]),(src[3][0],src[3][1]),(255,0,0),10)    
    cv2.line(line_image,(src[1][0],src[1][1]),(src[2][0],src[2][1]),(255,0,0),10)        
    
    dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                      [img_size[0]-offset, img_size[1]-offset], [offset, img_size[1]-offset]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(frame_img, M, img_size)
    
#    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))    
    
#    weighted = cv2.addWeighted(frame_img, 0.8, line_image, 1, 0) 
#    plt.imshow(cv2.cvtColor(weighted, cv2.COLOR_BGR2RGB))    
   
calibrate_camera()    

#test_image()