import cv2
import numpy as np


def has_background(image):

    """
    Check if the image has a significant background.
    Assumes that the background is white or nearly white.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    non_zero_count = np.count_nonzero(binary)

    return non_zero_count > 1000  # Adjust the threshold based on your image size


def detect_background(image_file):
    """
    Detect if the background is present or if it has been removed but objects are still present.
    """
    img_fs = image_file.read()
    np_ary = np.frombuffer(img_fs, np.uint8)
    img = cv2.imdecode(np_ary, cv2.IMREAD_COLOR)

    if img is None:
        return "Error: Image file could not be read."

    bg_present = has_background(img)
    if bg_present:
        return 250
    elif not bg_present:
        return 0
    else:
        return "Image is empty or does not fit expected criteria."
