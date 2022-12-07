import cv2
import os
import numpy as np


def read_img(img_path: str):
    """
    Load an image
    """
    if os.path.exists(img_path):
        return cv2.imread(img_path)
    else:
        raise FileNotFoundError(f"{img_path} not found")


def median_blur(img: np.array, k_size: int):
    """
    Apply median blur to an image
    """
    return cv2.medianBlur(img, k_size, 0)


def gaussian_blur(img: np.array, k_size: int):
    """
    Apply Gaussian blur to an image
    """
    return cv2.GaussianBlur(img, (k_size, k_size), 0)


def convert_bgr_to_yuv(img: np.array):
    """
    Convert BGR to a YUV
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)


def find_nssd(mask: np.array, cutout: np.array, template: np.array):
    """
    Compute the normalized sum of square difference with a mask
    """
    mask = np.float32(mask)
    cutout = np.float32(cutout)
    template = np.float32(template)
    numerator = np.sum(((cutout - template) * mask) ** 2)
    denominator = np.sqrt(np.sum((cutout * mask) ** 2) * np.sum((template * mask) ** 2))
    return numerator / denominator
