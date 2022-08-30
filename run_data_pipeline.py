#TODO: not done

from typing import Dict
from utilities.dir import create_directory
import argparse
import os
import numpy as np
import cv2
import json

parser = argparse.ArgumentParser(description='Data pipeline')
parser.add_argument('--data_dir', default=None, type=str, help='Path to image folder')
parser.add_argument('--new_data_dir', default=None, type=str, help='Path to save new data')
args = parser.parse_args()

if __name__ == '__main__':
    image_dir = os.path.join(args.data_dir, 'image')
    label_dir = os.path.join(args.data_dir, 'label')
    seg_dir = os.path.join(args.data_dir, 'segmentation_label')

    image_list = os.listdir(image_dir)
    for idx in range(len(image_list)):
        image_path = os.path.join(image_dir, image_list[idx])
        label_path = os.path.join(label_dir, image_list[idx].split('.')[0] + '.json')
        seg_path = os.path.join(seg_dir, image_list[idx].split('.')[0] + '.txt')
        with open(label_path) as f:
            annotations = json.load(f)
        cnt = [0 for _ in range(2)]
        for annotation in annotations:
            cnt[annotation['label'] == 107] += 1
        case_images = []
        if cnt[1] >= 1 and cnt[0] >= 1:
            case_images.append((image_path, label_path))