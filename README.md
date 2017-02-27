## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[image_carnd_advanced_lane]: ./output_images/carnd_advanced_lane.png "Project structure"
[image_load_calibration]: ./output_images/load_calibration.png "Load calibration points"
[image_calibration_verification_1]: ./output_images/calibration_verification_1.png "Calibration result"
[image_calibration_verification_2]: ./output_images/calibration_verification_2.png "Calibration result"
[image_calibration_verification_3]: ./output_images/calibration_verification_3.png "Calibration result"

[image_explorer_src_dst]: ./output_images/explorer_src_dst.png "Interactive UI for exploring src and dst points"


[image_birds_eye_view_verification1]: ./output_images/birds_eye_view_verification1.png "Birds-eye verification result"
[image_birds_eye_view_verification2]: ./output_images/birds_eye_view_verification2.png "Birds-eye verification result"
[image_birds_eye_view_verification3]: ./output_images/birds_eye_view_verification3.png "Birds-eye verification result"
[image_birds_eye_view_verification4]: ./output_images/birds_eye_view_verification4.png "Birds-eye verification result"
[image_birds_eye_view_verification5]: ./output_images/birds_eye_view_verification5.png "Birds-eye verification result"
[image_birds_eye_view_verification6]: ./output_images/birds_eye_view_verification6.png "Birds-eye verification result"


[image_threshold_verification1]: ./output_images/threshold_verification1.png "Threshold verification result"
[image_threshold_verification2]: ./output_images/threshold_verification2.png "Threshold verification result"
[image_threshold_verification3]: ./output_images/threshold_verification3.png "Threshold verification result"
[image_threshold_verification4]: ./output_images/threshold_verification4.png "Threshold verification result"
[image_threshold_verification5]: ./output_images/threshold_verification5.png "Threshold verification result"
[image_threshold_verification6]: ./output_images/threshold_verification6.png "Threshold verification result"

[image_histogram_peak]: ./output_images/histogram_peak.png "Histogram of peaks"

[image_polyfit_verification1]: ./output_images/polyfit_verification1.png "Polyfit - Verification"
[image_polyfit_verification2]: ./output_images/polyfit_verification2.png "Polyfit - Verification"
[image_polyfit_verification3]: ./output_images/polyfit_verification3.png "Polyfit - Verification"
[image_polyfit_verification4]: ./output_images/polyfit_verification4.png "Polyfit - Verification"
[image_polyfit_verification5]: ./output_images/polyfit_verification5.png "Polyfit - Verification"
[image_polyfit_verification6]: ./output_images/polyfit_verification6.png "Polyfit - Verification"

[image_overlay_result1]: ./output_images/overlay_result1.png "Overlay of the warped image to original image"

###Project Structure
The solution consists of one Jupyter Notebook and several custom python modules. Most of the image processing logics and lane finding algorithms are split in the several python modules and stored in the "lib" folder, and I used the Jupyter Notebook as a client application that leverages these modules. Here is the overview of the project structure: 
![alt text][image_carnd_advanced_lane]

- Jupyter Notebook file (P4_Advanced_Lane_Finding.ipynb): This is the entry point for the execution of the solution. This file is dependent on several custom modules as well as other python modules such as numpy and matplotlib.
- CameraCalibrator (lib/calibrator.py): This class is responsible for providing camera calibration functionalities.
- CarLane and Line (lib/carlane.py): CarLane and Line classes are responsible for computing the line detection and representing the left and right car lines.
- PerpsectiveTransformer (lib/perspective_transformer.py): PerspectiveTransformer class is responsible for generating the "birds-eye view" image
- ImageUtil (lib/imageutil.py): This class exposes several useful image manipulation methods
- VideoUtil (lib/videoutil.py): This class exposes several useful video manipulation methods

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the load_calibration_points method in the CameraCalibrator class.

Using the provided chessboard images, the load_calibration_points method went through each chessboard image and tried to find chessboard corners with the pattern size of (9, 6) using OpenCV's findChessboardCorners function. If all of the corners are found and they are placed in a certain order (row by row, left to right in every row), I store the return value from OpenCV's findChessboardCorners in `objpoints` which is an array of #d points in the real world space and `imgpoints` which is an array of 2d points in image plane.

Once the calibration points are loaded, I used calibrate_camera method in the CameraCalibrator class to compute the camera matrix and distortion coefficients. The calibrate_camera method leverages OpenCV's calibrateCamera function to compute the matrix and coefficients.

Finally, I provided the undistort method in the CameraCalibrator class which internally leverages OpenCV's undistort function, and I obtained these results:

- Loading calibration points
![alt text][image_load_calibration]

- Example of a distortion corrected calibration image
![alt text][image_calibration_verification_1]


###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like these examples:

- Example 1
![alt text][image_calibration_verification_2]

- Example 2
![alt text][image_calibration_verification_3]


####2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is found in the PerspectiveTransformer class. In the 8th code cell of the Jupyter Notebook, I created the explorer_perspective_transform_points method to explorer the various source and destination points with the UI as shown in the screenshot below:

![alt text][image_explorer_src_dst]


This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Here is a result of several warped images based on test images:

![alt text][image_birds_eye_view_verification1]
![alt text][image_birds_eye_view_verification2]
![alt text][image_birds_eye_view_verification3]
![alt text][image_birds_eye_view_verification4]
![alt text][image_birds_eye_view_verification5]
![alt text][image_birds_eye_view_verification6]

####3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for the generation of a thresholded binary image can be found in the binary_thresholded_image method in the ImageUtil class. From the 10th to 13th code cells of the Jupyter Notebook, I explored various ways to explorer the optimal way to identify car lane lines. I explored various techniques including color and gradient thresholds to generate a binary image, and focused on how to identify the white and yellow lines. Techniques I explored included:
- Sobel operators
- Color thresholds based on channels on color spaces (HLS, HSV, LAB, and LUV)

For the project video and challenge video, the combination of the L channel in the LUV color space and B channel in the LAB color space worked best for me. In summary:
- L channel threshold (215 ~ 255)
- B channel threshold (145 ~ 200)
- Output: combined binary of the L channel and B channel thresholds.

Here is the result of the generated binary images based on the warped images created from Step2.

![alt text][image_threshold_verification1]
![alt text][image_threshold_verification2]
![alt text][image_threshold_verification3]
![alt text][image_threshold_verification4]
![alt text][image_threshold_verification5]
![alt text][image_threshold_verification6]


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for identifying lane-line pixels can be found in the find_lane_pixels method of the CarLane class (lib/carlane.py). The code for calculating the polynomial fit can be found in the polyfit_lines method of the Line class (lib/carlane.py).  

The first step of identifying the lane-line pixels is to find the peak for the left and right car line by examining the histogram of the binary thresholded image.
![alt text][image_histogram_peak]

As shown on the above histogram, there were two peaks, one on the left for a left car lane and the other for the right car lane. The next step is to perform a window sliding technique to capture the pixels on the specific window slide until the window sliding is complete. The picture below shows on the region of boxes in each window slide for searching pixels within the boxes.

![alt text][image_polyfit_verification6]

When the above step is done using the find_lane_pixels method of the CarLane class, I passed the discovered pixels to the polyfit_lines method of the left and right car line instance in the Line class (lib/carlane.py) to calculate the polynomial fit for each car line. 


####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

As described in this article ["Radius of Curvature"](http://www.intmath.com/applications-differentiation/8-radius-curvature.php), the radius of curvature of the curve at a particular point is defined as the radius of the approximating circle. The formula for the radius of curvature at any point x for the curve y = f(x) is given by:
```python
	radius_of_curvature = ((1 + (2 * line_fit_cr[0] * y_eval * CarLane.ym_per_pix + line_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * line_fit_cr[0])
```
where 
```python
	line_fit_cr[0] = A and line_fit_cr[1] = B 
```	
where A and B are the first and second term in a second order polynomial curve such that:
```
	f(y) = A * y * y + B * y + C
```	

The position of the vehicle can be calculated using the following code:
```python
	vehicle_position = line_fit[0] * image_shape[0] ** 2 + line_fit[1] * image_shape[0] + line_fit[2]
```

The code for above steps are found in the polyfit_lines method in the Line class (lib/carlane.py).

Once both radius_of_curvature and vehicle_position are found for each car line, the CarLane class returns the final vehicle position and radius_of_curvature using the following method:
```python
    def vehicle_position(self, image_shape):
        """ Return the position of vehicle """
        return abs(image_shape[1] // 2 - ((self.left.line_base_pos + self.right.line_base_pos) / 2))

    def radius_of_curvature(self):
        """ Return the radius of curvature """
        return (self.left.radius_of_curvature + self.right.radius_of_curvature) // 2
``` 


####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The overlay_image method in the CarLane class (lib/carlane.py) generates the lane area by creating a polygon filled in green color and warping the polygon with the inverse matrix for the perspective transformation. Then, the transformed polygon is overlayed onto the original image.

I implemented the "console view" to show the overall pipeline of the image detection and the status of the pipeline. Here is an example of the console view:

![alt text][image_overlay_result1]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a video recording for the project video (./project_video.mp4):

[![Alt text](https://img.youtube.com/vi/c71xPQXt2cg/0.jpg)](https://www.youtube.com/watch?v=c71xPQXt2cg)


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I observed in the challenge_video.mp4 and harder_challenge_video_result.mp4 that it is difficult to identify car lines especially the white car lines with the binary threshold settings configured for the project video.

I used the following techniques to make the pipeline more robust:
- I kept the previous line pixels for each car line. When no pixels were found for the current image frame, I used the previous line pixels to recover from the failure.

Here are some ideas to make the pipeline more robust:
- Capture the brightness level of each frame, and depending on the brightness level, I may use a different binary threshold technique, i.e. using HSV filter in addition to existing LUV and LAB filters.
- Validation of the line pixels and line fitting curve based on the history of the line pixels and fitting values. When there is a sudden change to the distribution of line pixels and fitting values compared to the previous value, I may discard the current line pixels and use the previously captured line pixels or capture line pixels using different binary threshold technique.
