import cv2
import numpy as np
from utils import convert_bgr_to_yuv

class Color:
    """
    A class that defines colors
    """

    def __init__(self, name: str, bgr: np.array):
        self.name = name
        self.bgr = bgr
        self.yuv = self.convert_bgr_to_yuv()

    def convert_bgr_to_yuv(self):
        """
        Convert a BGR-formatted color to a YUV-formatted color
        """
        bgr_array = np.array([[self.bgr]])
        return convert_bgr_to_yuv(bgr_array)[0][0]


# Colors of the background of road signs
yellow_sign = Color("yellow_sign", np.uint8([51, 208, 255]))
red_sign = Color("red_sign", np.uint8([46, 25, 166]))
# white_sign = Color("white", np.uint8([255, 255, 255]))

# Colors of the traffic lights
red_light = Color("red_light", np.uint8([0, 0, 255]))
green_light = Color("green_light", np.uint8([255, 255, 0]))
yellow_light = Color("yellow_light", np.uint8([128, 255, 255]))
