import cv2
import numpy as np
import os
from typing import List
import segmentation_refinement as refine
import json
import random

def read_image(image_path: str):
    return cv2.imread(image_path)

def read_bbox(bbox_path: str):
    with open(bbox_path) as f:
        return json.load(f)

IMAGE_DIR = 'data/vaipe/public_test/image'
BBOX_DIR = 'data/vaipe/public_test/label'
SEG_DIR = 'data/vaipe/public_test/segmentation'
refiner = refine.Refiner(device='cuda:0') # device can also be 'cpu'

def get_contour_coordinates(contour):
    approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True)
    return approx.ravel()

def single_image(image):
    H, W = image.shape[:2]
    mask = np.zeros([H,W], dtype=np.uint8)
    mask.fill(0) # or img[:] = 255

    border = int(min(H, W) * 0.05)
    mask[border:-border, border:-border] = 255

    # Smaller L -> Less memory usage; faster in fast mode.
    # Fast - Global step only.
    output = refiner.refine(image, mask, fast=False, L=150)
    output[output > 130] = 255
    output[output <= 130] = 0
    _, threshold = cv2.threshold(output, 127, 255, 0)
    
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_sizes= [(cv2.contourArea(cnt), cnt) for cnt in contours]
    if len(contours_sizes) == 0:
        return np.array([5, 5, W-1, 5, W-1, H-1, 5, H-1])

    biggest_contour = max(contours_sizes, key=lambda x: x[0])[1]
    coordinates = get_contour_coordinates(biggest_contour)
    return coordinates

def draw_contour(image, coordinates):
    a = []
    color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    for i in range(0, len(coordinates), 2):
        x = coordinates[i]
        y = coordinates[i + 1]
        a.append((x, y))
    a = np.array(a)
    cv2.drawContours(image, [a], 0, color[random.randint(0, 2)], 1)
    return image

def get_segmentation(image: np.array, annotations: List, normalize = False):
    H, W = image.shape[:2]
    coordinates_segment = []
    for annotation in annotations:
        border = 0
        x = max(0, annotation['x'] - border)
        y = max(0, annotation['y'] - border)
        w = min(annotation['w'] + border, W - 1)
        h = min(annotation['h'] + border, H - 1)

        crop_image = image[y:y+h, x:x+w]
        # cv2.imshow('image', crop_image)
        # cv2.waitKey(0)
        coordinates = single_image(crop_image)
        if len(coordinates) == 0:
            continue
        coordinates = coordinates.astype(np.float32)
        for i in range(0, len(coordinates), 2):
            coordinates[i] += x
            coordinates[i+1] += y
            if normalize:
                coordinates[i] /= W
                coordinates[i+1] /= H
                coordinates[i] = round(coordinates[i], 5)
                coordinates[i + 1] = round(coordinates[i + 1], 5)
        # print(coordinates)
        coordinates_segment.append(coordinates)
    return coordinates_segment

def write_annotation_txt(file_txt: str, coordinates_list):
    f = open(file_txt, 'w')
    for coordinates in coordinates_list:
        coordinates = [str(item) for item in coordinates.tolist()]
        f.write(str(0) + ' ' + ' '.join(coordinates) + '\n')
    f.close()

if __name__ == '__main__':
    image_list = os.listdir(IMAGE_DIR)
    wrong_image = []
    for id, image_name in enumerate(image_list):
        print(f'[{id + 1}/{len(image_list)}] Processing {image_name} ...')
        image = read_image(os.path.join(IMAGE_DIR, image_name))
        annotations = read_bbox(os.path.join(BBOX_DIR, image_name.split('.')[0] + '.json'))

        coordinates_list = get_segmentation(image, annotations, True)
        if len(coordinates_list) == 0:
            wrong_image.append(image_name)
            print(f'Something wrong with this shit, {image_name}')
        write_annotation_txt(os.path.join(SEG_DIR, image_name.split('.')[0] + '.txt'), coordinates_list)

    print(wrong_image)


    # image_name = 'VAIPE_P_167_10.jpg'
    # image = read_image(os.path.join(IMAGE_DIR, image_name))
    # annotations = read_bbox(os.path.join(BBOX_DIR, image_name.split('.')[0] + '.json'))

    # coordinates_list = get_segmentation(image, annotations)
    # for coordinates in coordinates_list:
    #     draw_contour(image, coordinates)

    # cv2.imshow('image', image)
    # cv2.waitKey(0)