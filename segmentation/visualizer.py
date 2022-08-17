import cv2
import matplotlib.pyplot as plt
from .process import get_rectangle_from_points, get_all_mask_points
import numpy as np

def draw_rectangle(image, masks):
    copy = image.copy()
    for mask in masks:
        points = get_all_mask_points(mask)
        xmin, ymin, xmax, ymax = get_rectangle_from_points(points)
        copy = cv2.rectangle(copy, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    plt.imshow(copy)


def draw_mask(image, masks):
    clone = image.copy()
    # plt.imshow(clone)
    color_mask = np.zeros(clone.shape).astype('float32')
    for mask in masks:
        points = get_all_mask_points(mask)
        xmin, ymin, xmax, ymax = get_rectangle_from_points(points)
        color_mask[mask] = [0, 0, 255]
        clone = cv2.rectangle(clone, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        
    out = cv2.addWeighted(clone, 0.98, color_mask, 0.02, 0)
    plt.imshow(out)