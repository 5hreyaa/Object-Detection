import cv2
import numpy as np

def preprocess_image(image):
    # Handle different resolutions
    max_size = 1280
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image