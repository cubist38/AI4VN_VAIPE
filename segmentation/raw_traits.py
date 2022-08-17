import numpy as np
import cv2
from typing import Dict
from semantic import get_mask

def get_area_2d(mask: np.array):
    pixels = cv2.findNonZero(mask)
    return len(pixels)
    
def get_diameter(mask: np.array):
    pixels = cv2.findNonZero(mask)
    pixels = pixels.squeeze(1)

    def _find_furthest_pair_point(points: np.array, axis: int = 0):
        """
        Find a furthest pair point over an given axis

        Parameters
        ----------
        points: ndarray
            an array of the coordinate of points

        axis: int
            Axis or axes along which finds the furthest pair point
        ----------
        Returns
        ----------
        furthest_pair_point: tuple of ndarray
            a tuple of coordinate of the furthest pair point
        ----------
        """
        min_point, max_point = points[..., axis].min(), points[..., axis].max()
        min_point_ids, max_point_ids = np.where(points[..., axis] == min_point)[0], \
                                       np.where(points[..., axis] == max_point)[0]

        if points[max_point_ids[0]][1 - axis] >= points[min_point_ids[0]][1 - axis]:
            return points[min_point_ids[0]], points[max_point_ids[-1]]
        elif points[max_point_ids[0]][1 - axis] < points[min_point_ids[0]][1 - axis]:
            return points[min_point_ids[-1]], points[max_point_ids[0]]

    leftmost_point, rightmost_point = _find_furthest_pair_point(pixels, 0)
    topmost_point, bottommost_point = _find_furthest_pair_point(pixels, 1)
    dist = lambda x, y: np.linalg.norm(x - y)
    diameter = max(dist(rightmost_point, leftmost_point), dist(topmost_point, bottommost_point))
    
    return diameter

def get_maximum_diameter(masks: np.array):
    max_diameter = 0
    for mask in masks:
        max_diameter = max(max_diameter, get_diameter(np.float32(mask)))
    return max_diameter

def get_maximum_area_2d(masks: np.array):
    max_area = 0
    for mask in masks:
        max_area = max(max_area, get_area_2d(np.float32(mask)))
    return max_area

def get_volume(depth: np.array, mask: np.array):
    depth = depth * mask.astype(int)
    indices = np.where(depth > 0)
    volume = 0
    for y, x in zip(indices[0], indices[1]):
        volume += 1400 - depth[y][x]
    return volume

def get_average_volume(depth_file: str, masks: np.array):
    depth = cv2.imread(depth_file, -1)
    volume = 0
    for mask in masks:
        volume += get_volume(depth, mask)
    volume /= masks.shape[0]
    return volume

# =================================================
# BASE ON SEMANTIC SEGMENTAION
# =================================================

def get_area_2d_total(config: Dict, image_file: str):
    img = cv2.imread(image_file)
    mask = get_mask(config, img)
    return get_area_2d(mask)

def get_coverage_ratio(config: Dict, image_file: str) -> float:
    img = cv2.imread(image_file)

    h, w, _ = img.shape
    mask = get_mask(config, img)
    region_area = h * w
    plant_area = mask.sum()
    return plant_area / region_area

def get_total_volume(config: Dict, image_file: str, depth_file: str):
    rgb = cv2.imread(image_file)
    depth = cv2.imread(depth_file, -1)
    mask = get_mask(config, rgb)
    depth = depth * mask.astype(int)
    indices = np.where(depth > 0)
    total_volume = 0
    for y, x in zip(indices[0], indices[1]):
        total_volume += 1400 - depth[y][x]
    return total_volume

def get_height(config: Dict, image_file: str, depth_file: str):
    rgb = cv2.imread(image_file)
    depth = cv2.imread(depth_file, -1)
    mask = get_mask(config, rgb)
    depth = depth * mask.astype(int)
    indices = np.where(depth > 0)
    average_height = 0
    n = len(indices[0])
    for y, x in zip(indices[0], indices[1]):
        average_height += depth[y][x] / n
    return average_height