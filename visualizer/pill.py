from .config import *

from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import imutils

def down_scale_image(image: np.array) -> np.array:
    (h, w) = image.shape[:2]
    if h > w:
        return imutils.resize(image, height=MAX_IMAGE_SIZE)
    else:
        return imutils.resize(image, width=MAX_IMAGE_SIZE)

def apply_bbox(image: np.array, bbox_list: List[Dict], down_scale: bool = True) -> np.array:
    for bbox in bbox_list:
        xmin, ymin = bbox['x'], bbox['y']
        xmax, ymax = bbox['x'] + bbox['w'], bbox['y'] + bbox['h']
        label = str(bbox['label'])

        # bounding box
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR, THICKNESS)

        # text
        (w, h), _ = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
        image = cv2.rectangle(image, (xmin, ymin - 30), (xmin + w, ymin), COLOR, -1)
        image = cv2.putText(image, label, (xmin, ymin-10), FONT, FONT_SCALE, (255, 255, 255), THICKNESS, LINE)

    if down_scale:
        image = down_scale_image(image)

    return image