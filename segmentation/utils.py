import os
import json
import cv2
import os
import numpy as np
from detectron2.structures import BoxMode
import yaml

def create_directory(directory):
    dirs = directory.split('/')
    path = ''
    for folder in dirs:
        path = os.path.join(path, folder)
        if not os.path.exists(path):
            os.makedirs(path)

def read_config(file_path: str):
    with open(file_path) as f:
        return yaml.safe_load(f)

def merge_json(json_set):
    total_imgs_anns = []
    for json_name in json_set:
        json_file = os.path.join('data/annotation', json_name + '.json')
        with open(json_file) as f:
            imgs_anns = json.load(f)
        total_imgs_anns += imgs_anns
    print(len(total_imgs_anns))
    return total_imgs_anns

def get_date(image_name):
    return image_name[14:18]

def get_lettuce_dicts(imgs_anns):
    dataset_dicts = []
    for idx, v in enumerate(imgs_anns):
        record = {}
        
        filename = v["rgb_file"]
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        annos = v["segmentation"]
        objs = []
        for anno in annos:
            px = anno["x_point"]
            py = anno["y_point"]
            
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

