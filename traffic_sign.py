import cv2
import numpy as np
from color import Color, yellow_sign, red_sign
from utils import gaussian_blur

class TrafficSign:
    """
    A class tat defines a traffic sign
    """

    def __init__(self, name: str, color: Color, k_size: int = 5):
        self.name = name
        self.k_size = k_size
        self.img = cv2.imread(f"traffic_sign_templates/{name}.png")
        self.img = gaussian_blur(self.img, k_size)
        self.shape = self.img.shape
        self.color = color
        self.bgr = color.bgr
        self.yuv = color.yuv


def create_traffic_sign_dict(k_size: int = 5):
    """
    Create a traffic sign dictionary, categorized by its color.
    A total of 13 signs (11 yellow, 2 red) are used for detection.
    No white sign is used due to its bad behavior
    """

    yellow_sign_list = [ 
        TrafficSign("addedLane", yellow_sign, k_size),
        TrafficSign("dip", yellow_sign, k_size),
        TrafficSign("intersection", yellow_sign, k_size),
        TrafficSign("laneEnds", yellow_sign, k_size),
        TrafficSign("merge", yellow_sign, k_size),
        TrafficSign("pedestrianCrossing", yellow_sign, k_size),
        TrafficSign("signalAhead", yellow_sign, k_size),
        TrafficSign("stopAhead", yellow_sign, k_size),
        TrafficSign("turnLeft", yellow_sign, k_size),
        TrafficSign("turnRight", yellow_sign, k_size),
        TrafficSign("yieldAhead", yellow_sign, k_size),
    ]

    red_sign_list = [
        TrafficSign("stop", red_sign, k_size),
        TrafficSign("yield", red_sign, k_size),
    ]


    traffic_sign_dict = {
        yellow_sign: yellow_sign_list,
        red_sign: red_sign_list,
    }

    return traffic_sign_dict


traffic_sign_list = []
traffic_sign_dict = create_traffic_sign_dict()
for sign_list in traffic_sign_dict.values():
    for sign in sign_list:
        traffic_sign_list.append(sign.name)
        