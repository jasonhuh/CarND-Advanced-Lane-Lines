# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 12:35:06 2016

@author: macuser
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('test4.jpg')
thresh = (180, 255)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
binary = np.zeros_like(gray)
binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1

#plt.imshow(binary, cmap='gray')
#plt.imshow(image[:,:,0], cmap='gray')

R = image[:,:,0]
G = image[:,:, 1]
B = image[:,:, 2]


thresh = (200, 255)
binary = np.zeros_like(R)
binary[(R > thresh[0]) & (R <= thresh[1])] = 1

#plt.imshow(binary, cmap='gray')

hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
H = hls[:,:,0]
L = hls[:,:,1]
S = hls[:,:,2]

thresh = (90, 255)
binary = np.zeros_like(S)
binary[(S > thresh[0]) & (S <= thresh[1])] = 1

plt.imshow(binary, cmap='gray')