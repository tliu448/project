import cv2
import os
import numpy as np
from utils import median_blur, convert_bgr_to_yuv, read_img
from color import Color, red_light, green_light, yellow_light


class TrafficLightDetector:
    """
    A class that detects traffic light in an image
    """

    def __init__(self, k_size: int, cutout: float, min_radius: float, max_radius: float, yuv_error: np.array, dilation: int, debug: bool):
        self.k_size = k_size # the kernel size used in smoothing
        self.cutout = cutout # the % of upper part of the image to keep
        self.min_radius = min_radius # the minimum radius allowed in cv2.HoughCircles
        self.max_radius = max_radius # the maximum radius allowed in cv2.HoughCircles
        self.yuv_error = yuv_error # the error that determines the range of YUV to look for
        self.dilation = dilation # the size of dilation
        self.debug = debug 


    def cutout_img(self, img: np.array):
        """
        Cut the image to keep only the upper part of an image, where traffic lights are usually found
        """
        h = img.shape[0]
        return img[:round(h*self.cutout), :, :]


    def detect_color(self, img: np.array, color: Color):
        """
        Detect a given color in the image.
        Return a boolean matrix where 1 denotes the presence of the color in that pixel and 0 otherwise
        """
        
        img = convert_bgr_to_yuv(img)
        background_yuv = np.float32(color.yuv)
        lower_bound = background_yuv - self.yuv_error
        upper_bound = background_yuv + self.yuv_error
        matched = np.logical_and(lower_bound < img, img < upper_bound).all(axis=2)

        matched = np.uint8(matched)
        kernel = np.ones((self.dilation, self.dilation), np.uint8)
        matched = cv2.dilate(matched, kernel, iterations=1)
        matched = cv2.erode(matched, kernel, iterations=1)

        if self.debug:
            cv2.imwrite(f"background_matching_{color.name}.png", matched * 255)

        return matched


    def detect_circles(self, matched: np.array, color: Color):
        """
        Detect circles in the boolean matrix of a color found by self.detect_color()
        """
        h, w = matched.shape
        size = np.mean([h, w])
        min_r = round(size*self.min_radius)
        max_r = round(size*self.max_radius)
        min_r = max(2, min_r)
 
        circles = cv2.HoughCircles(matched * 255, cv2.HOUGH_GRADIENT, 1, 10, param1=40, param2=5, minRadius=min_r, maxRadius=max_r)
        
        if circles is None:
            if self.debug:
                print(f"No circle is found in this color: {color.name}")
            return

        circles = circles[0]
        circles = np.uint16(np.around(circles))

        bgr = cv2.cvtColor(matched, cv2.COLOR_GRAY2BGR)
        if self.debug:
            for c in circles:
                bgr = cv2.circle(bgr, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.imwrite(f"circle_detecting_{color.name}.png", bgr)

        return circles

    
    def detect_traffic_light(self, img: np.array):
        """
        Detect traffic light using both color and circles as clues
        """
        
        green_matched = self.detect_color(img, green_light)
        red_matched = self.detect_color(img, red_light)
        yellow_matched = self.detect_color(img, yellow_light)

        green_circles = self.detect_circles(green_matched, green_light)
        red_circles = self.detect_circles(red_matched, red_light)
        yellow_circles = self.detect_circles(yellow_matched, yellow_light)

        if green_circles is not None:
            return "go", green_circles
        elif red_circles is not None:
            return "stop", red_circles
        elif yellow_circles is not None:
            return "warning", yellow_circles
        else:
            return "", None

    
    @staticmethod
    def draw_boundary(img: np.array, circle: np.array):
        """
        Draw a boundary box to describe the object found
        """
        x, y, r = circle
        img = cv2.rectangle(img, (x-r,y-r), (x+r,y+r), (255,0,255), 1)
        return img


    @staticmethod
    def draw_text(img: np.array, text: str, position: tuple):
        """
        Draw text on the upper left corner of the boundary
        """
        
        img = cv2.putText(img, f"{text} {position}", position, color=(255,0,255), fontFace=0, fontScale=0.6, thickness=1)
        return img

    
    def draw_all_boundary_and_text(self, img: np.array, matched_color: str, circle: np.array):
        """
        Draw both boundary and text
        """
        x, y, r = circle
        img = self.draw_boundary(img, circle)
        img = self.draw_text(img, matched_color, (x-r, y-r))
        return img


    def run(self, img_file: str, output_dir: str):
        """
        Combine everything together.
        Take an input image, find traffic lights, draw boundary and text, write to file
        """
        img_orig = read_img(img_file)
        img = self.cutout_img(img_orig)
        img = median_blur(img, self.k_size)

        filename = os.path.basename(img_file)
        matched_annotation, matched_circles = self.detect_traffic_light(img)
        if matched_annotation:
            for matched_circle in matched_circles:
                matched_circle = np.int16(matched_circle)
                img_orig = self.draw_all_boundary_and_text(img_orig, matched_annotation, matched_circle)
            number_of_matches = len(matched_circles)
        else:
            matched_annotation = ""
            number_of_matches = 0

        cv2.imwrite(os.path.join(output_dir, filename), img_orig)
        return matched_annotation, number_of_matches
