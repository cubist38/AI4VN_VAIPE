import cv2
import json

def read_image(image_path: str):
    return cv2.imread(image_path)

def read_bbox(bbox_path: str):
    with open(bbox_path) as f:
        return json.load(f)