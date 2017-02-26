import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageUtil():

    @staticmethod
    def image_console(mainScreen, screen1, screen2, screen3, text1, text2):
        # middle panel text example
        # using cv2 for drawing text in diagnostic pipeline.
        font = cv2.FONT_HERSHEY_COMPLEX
        middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
        cv2.putText(middlepanel, text1, (30, 60), font, 1, (255, 0, 0), 2)
        cv2.putText(middlepanel, text2, (30, 90), font, 1, (255, 0, 0), 2)

        # assemble the screen example
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        canvas[0:720, 0:1280] = mainScreen
        canvas[0:360, 1280:1760] = cv2.resize(screen1, (480, 360), interpolation=cv2.INTER_AREA)
        canvas[360:720, 1280:1760] = cv2.resize(screen2, (480, 360), interpolation=cv2.INTER_AREA)
        canvas[720:1080, 1280:1760] = cv2.resize(screen3, (480, 360), interpolation=cv2.INTER_AREA)
        canvas[720:840, 0:1280] = middlepanel
        return canvas

    """ Binary Threshold """
    @staticmethod
    def visualize_binary_thresholded_image(img, b_binary, l_binary, combined_binary):
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey='col', sharex='row', figsize=(24, 4))
        f.tight_layout()

        ax1.set_title('Warped Image', fontsize=16)
        ax1.imshow(img)

        ax2.set_title('B threshold', fontsize=16)
        ax2.imshow(b_binary, cmap='gray')

        ax3.set_title('L threshold', fontsize=16)
        ax3.imshow(l_binary, cmap='gray')

        ax4.set_title('Combined thresholds', fontsize=16)
        ax4.imshow(combined_binary, cmap='gray')

    @staticmethod
    def binary_thresholded_image(img, visualize=False, b_thresh_min=145, b_thresh_max=200, \
                                 l_thresh_min=215, l_thresh_max=255):
        l_channel = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)[:, :, 0]  # Detect white lines
        b_channel = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, 2]  # Detect yellow lines

        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

        b_binary = np.zeros_like(b_channel)
        b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

        combined_binary = np.zeros_like(b_binary)
        combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

        if visualize: ImageUtil.visualize_binary_thresholded_image(img, b_binary, l_binary, combined_binary)

        return combined_binary