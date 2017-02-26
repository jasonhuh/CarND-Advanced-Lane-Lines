import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageUtil():

    @staticmethod
    def image_console(mainScreen, sc1, sc2, sc3, sc4, text_arr):
        # middle panel text example
        # using cv2 for drawing text in diagnostic pipeline.
        font = cv2.FONT_HERSHEY_COMPLEX
        #middlepanel = np.zeros((120, 1280, 3), dtype=np.uint8)
        textpanel = np.zeros((240, 640, 3), dtype=np.uint8)
        text_pos = 30
        for text in text_arr:
            cv2.putText(textpanel, text, (30, text_pos), font, 1, (255, 0, 0), 2)
            text_pos += 30
            #if text2 is not None: cv2.putText(textpanel, text2, (30, 90), font, 1, (255, 0, 0), 2)

        # assemble the screen example
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        canvas[0:720, 0:1280] = mainScreen

        #canvas[0:240, 1280:1920] = textpanel #cv2.resize(sc1, (640, 240), interpolation=cv2.INTER_AREA)
        canvas[0:240, 1280:1600] = cv2.resize(sc2, (320, 240), interpolation=cv2.INTER_AREA)
        canvas[0:240, 1600:1920] = cv2.resize(sc4, (320, 240), interpolation=cv2.INTER_AREA)

        canvas[240:480, 1280:1920] = textpanel #cv2.resize(sc1, (640, 240), interpolation=cv2.INTER_AREA)
        # canvas[240:480, 1280:1600] = cv2.resize(sc2, (320, 240), interpolation=cv2.INTER_AREA)
        # canvas[240:480, 1600:1920] = cv2.resize(sc4, (320, 240), interpolation=cv2.INTER_AREA) * 4
        #if sc7 is not None: canvas[600:1080, 1280:1920] = cv2.resize(sc7, (640, 480), interpolation=cv2.INTER_AREA) * 4

        canvas[720:1080, 0:1280] = cv2.resize(sc3, (1280, 360), interpolation=cv2.INTER_AREA) * 4
        #canvas[840:1080, 0:320] = textpanel
        #if sc5 is not None: canvas[840:1080, 0:320] = cv2.resize(sc5, (320, 240), interpolation=cv2.INTER_AREA)
        # if sc6 is not None: canvas[840:1080, 320:640] = cv2.resize(sc6, (320, 240), interpolation=cv2.INTER_AREA)
        # if sc9 is not None: canvas[840:1080, 640:960] = cv2.resize(sc9, (320, 240), interpolation=cv2.INTER_AREA)
        # if sc8 is not None: canvas[840:1080, 960:1280] = cv2.resize(sc8, (320, 240), interpolation=cv2.INTER_AREA)

        return canvas

    """ Binary Threshold """
    @staticmethod
    def binary_thresholded_image(img, b_thresh_min=145, b_thresh_max=200, \
                                 l_thresh_min=215, l_thresh_max=255):
        l_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:, :, 0]  # Detect white lines
        b_channel = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, 2]  # Detect yellow lines

        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

        b_binary = np.zeros_like(b_channel)
        b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

        combined_binary = np.zeros_like(b_binary)
        combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

        return combined_binary, (l_binary, b_binary)