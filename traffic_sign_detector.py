import cv2
import os
import numpy as np
from color import Color
from traffic_sign import TrafficSign, create_traffic_sign_dict
from typing import List, Dict
from utils import convert_bgr_to_yuv, find_nssd, read_img, gaussian_blur


class TrafficSignDetector:
    """
    A class that detects traffic sign in an image
    """

    def __init__(self, k_size: int, yuv_error: np.array, matching_lower_bound: float, matching_upper_bound: float, dilation: int, debug: bool):
        
        self.k_size = k_size # the kernel size used in smoothing
        self.yuv_error = yuv_error # the error that determines the range of YUV to look for
        self.matching_lower_bound = matching_lower_bound # the lower bound that determines whether a matching percentage is accepted
        self.matching_upper_bound = matching_upper_bound # the upper bound that determines whether a matching percentage is accepted
        self.dilation = dilation # the size of dilation
        self.debug = debug
        self.traffic_sign_dict = create_traffic_sign_dict(self.k_size)


    def detect_color(self, img: np.array, color: Color):
        """
        Detect a given color in the image.
        If a color is detected, find the connected component of this color with the largest area
        Return:
        - the matched color
        - a boolean matrix where 1 denotes the presence of the color in that pixel and 0 otherwise
        - the boundary of the largest connected component of the detected color
        """
        
        img = convert_bgr_to_yuv(img)
        background_yuv = np.float32(color.yuv)
        lower_bound = background_yuv - self.yuv_error
        upper_bound = background_yuv + self.yuv_error 
        matched = np.logical_and(lower_bound < img, img < upper_bound).all(axis=2)

        h, w = matched.shape
        number_of_pixels = h * w
        matched_ratio = np.sum(matched) / number_of_pixels
        
        if matched_ratio < self.matching_lower_bound:

            if self.debug:
                print(f"Not enough pixels matching the color {color.name}")

            return "", matched, np.nan

        matched = np.uint8(matched)
        kernel = np.ones((self.dilation, self.dilation), np.uint8)
        matched = cv2.dilate(matched, kernel, iterations=1)
        matched = cv2.erode(matched, kernel, iterations=1)

        if self.debug:
            cv2.imwrite(f"background_matching_{color.name}.png", matched * 255)

        stats  = cv2.connectedComponentsWithStats(matched)[2]
        sorted_stats = stats[stats[:, -1].argsort()]

        matched_area = sorted_stats[-2][-1]
        matched_percentage = matched_area / number_of_pixels

        if (matched_percentage > self.matching_upper_bound) or (matched_percentage < self.matching_lower_bound):
            if self.debug:
                print(f"{color.name} is not a matching color")
            return "", matched, np.nan

        boundary = sorted_stats[-2][:-1]

        return color.name, matched, boundary


    @staticmethod
    def calculate_loss(img: np.array, matched: np.array, boundary: np.array, templates: List[np.array]):
        """
        Compute the normalized sum of square difference between a window of the image to each of the template
        """
        x, y, w, h = boundary
        cutout = img[y:(y+h), x:(x+w), :]
        cutout_mono = cv2.cvtColor(cutout, cv2.COLOR_BGR2GRAY)
        mask = matched[y:(y+h), x:(x+w)]
        loss = []
        for template in templates:
            resized = cv2.resize(template, (w,h))
            resized_mono = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            loss.append(find_nssd(mask, cutout_mono, resized_mono))
        return loss


    def detect_traffic_sign(self, img: np.array):
        """
        Detect traffic sign through color detection and loss analysis
        """

        output = ()
        for color, sign_list in self.traffic_sign_dict.items():
            is_matched, matched, boundary = self.detect_color(img, color)
            if is_matched:
                template_list = [sign.img for sign in sign_list]
                loss_list = self.calculate_loss(img, matched, boundary, template_list)

                if self.debug:
                    loss_dict = {sign.name: loss for sign, loss in zip(sign_list, loss_list)}
                    print(f"{color.name} is a matching color with the following loss: {loss_dict}")

                matched_index = np.argmin(loss_list)
                matched_sign = sign_list[matched_index]
                output = (matched_sign.name, boundary)
                break
                
            else:
                if self.debug:
                    print(f"{color.name} is not a matching color")

        if not bool(output) and self.debug:
            print("No traffic sign is found for this image")

        return output


    @staticmethod
    def draw_boundary(img: np.array, boundary: np.array):
        """
        Draw a boundary box to describe the object found
        """
        x, y, w, h = boundary
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        return img


    @staticmethod
    def draw_text(img: np.array, text: str, position: tuple):
        """
        Draw text on the upper left corner of the boundary
        """
        img = cv2.putText(img, f"{text} {position}", position, color=(0,255,0), fontFace=0, fontScale=0.6, thickness=1)
        return img


    def draw_all_boundary_and_text(self, img: np.array, matched_sign: str, boundary: np.array):
        """
        Draw both boundary and text
        """
        x, y, _, _ = boundary
        img = self.draw_boundary(img, boundary)
        img = self.draw_text(img, matched_sign, (x, y))
        return img


    def run(self, img_file: str, output_dir: str):
        """
        Put everything together
        Take an input image, find traffic sign, draw boundary and text, write to file
        """
        img = read_img(img_file)
        blurred_img = gaussian_blur(img, self.k_size)
        output = self.detect_traffic_sign(blurred_img)

        if output:
            matched_sign, boundary = output
            output_img = self.draw_all_boundary_and_text(img, matched_sign, boundary)
            x, y, _, _ = boundary
        else:
            output_img = img
            matched_sign = ""
            x = np.nan
            y = np.nan

        filename = os.path.basename(img_file)
        cv2.imwrite(os.path.join(output_dir, filename), output_img)
        return matched_sign, (x, y)
