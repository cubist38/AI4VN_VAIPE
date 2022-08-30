from typing import Dict
import shutil
import cv2
import json
import os
from utilities.dir import create_directory

def crop_bbox_images(detection_results: Dict, crop_cfg: Dict):
    '''
        Crop images based on bounding box, using in classifier.
    '''
    print('Running cropping ...', end = ' ')
    shutil.rmtree(crop_cfg['crop_img_dir'], ignore_errors=True)
    create_directory(crop_cfg['crop_img_dir'])
    crop_detection_map = {}
    for image_path, boxes in detection_results.items():
        image_name = image_path.split('/')[-1]
        img = cv2.imread(image_path)
        for id, box in enumerate(boxes):
            x_min, y_min, x_max, y_max, class_id, confidence_score = box
            crop_img = img[y_min:y_max, x_min:x_max]
            crop_img_name = str(id) + '_' + image_name
            crop_detection_map[crop_img_name] = {
                'image_id': image_name,
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
            }
            cv2.imwrite(os.path.join(crop_cfg['crop_img_dir'], crop_img_name), crop_img)
    
    with open(crop_cfg['crop_detection_map'], "w") as f:
        json.dump(crop_detection_map, f)
    print('Done!')