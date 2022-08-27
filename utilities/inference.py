from typing import Dict

import albumentations as A
from ensemble_boxes import *

import cv2
import os

# ====================== AUGMENTATION ==================== #

def augment_image(img , image_name: str, augment_folder: str):
    transforms_1 = A.Rotate(always_apply = False, p = 1.0, limit = (180, 180), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, method='largest_box', crop_border=False)
    transformed_1 = transforms_1(image=img)
    transformed_image_1 = transformed_1["image"]
    cv2.imwrite(os.path.join(augment_folder, image_name.split('.')[0] + '_1.jpg'), transformed_image_1)
    #------------------------------------------------------------------------------------------------------
    transforms_2 = A.RandomBrightness(always_apply=False, p=1.0, limit= (0.2, 0.2))
    transformed_2 = transforms_2(image=img)
    transformed_image_2 = transformed_2["image"]
    cv2.imwrite(os.path.join(augment_folder, image_name.split('.')[0] + '_2.jpg'), transformed_image_2)
    #------------------------------------------------------------------------------------------------------
    transforms_3 = A.RandomBrightness(always_apply=False, p=1.0, limit= (-0.2, -0.2))
    transformed_3 = transforms_3(image=img)
    transformed_image_3 = transformed_3["image"]
    cv2.imwrite(os.path.join(augment_folder, image_name.split('.')[0] + '_3.jpg'), transformed_image_3)
    #------------------------------------------------------------------------------------------------------

def revert_from_Rotate180(bbox, Width, Height, conf, class_id):
    xmin = Width - bbox[2]
    ymin = Height - bbox[3]
    xmax = Width - bbox[0]
    ymax = Height - bbox[1]
    return [xmin, ymin, xmax, ymax, conf, class_id]

def normalized_bbox_coordinate(boxes, Image_Width, Image_Height):
    normalized_boxes = []
    for bbox in boxes:
        normalized_xmin = bbox[0]/Image_Width
        normalized_ymin = bbox[1]/Image_Height
        normalized_xmax = bbox[2]/Image_Width
        normalized_ymax = bbox[3]/Image_Height
        normalized_bbox = [normalized_xmin, normalized_ymin, normalized_xmax, normalized_ymax]
        normalized_boxes.append(normalized_bbox)
    return normalized_boxes


def normalized_to_real(normalized_bboxes, Image_Width, Image_Height):
    real_bboxes = []
    for normalized_bbox in normalized_bboxes:
        real_xmin = normalized_bbox[0]*Image_Width
        real_ymin = normalized_bbox[1]*Image_Height
        real_xmax = normalized_bbox[2]*Image_Width
        real_ymax = normalized_bbox[3]*Image_Height
        real_bbox = [int(real_xmin), int(real_ymin), int(real_xmax), int(real_ymax)]
        real_bboxes.append(real_bbox)
    
    return real_bboxes

def mycrop(img, x_min, y_min, x_max, y_max, offset = 10):
    if x_min - offset > 0:
        x_min = x_min - offset
    if y_min - offset > 0:
        y_min = y_min - offset
    
    return img[y_min : y_max, x_min : x_max]
    
# ====================== MAPPING ==================== #

# This is the function which maps from text to vaipe's label.
def find_vaipe_label(label_drugname: Dict, text):
    if text not in label_drugname:
        return None
    return label_drugname[text]

# We should pass the ocr_output_dict to this function to have the mapping.
def text_to_vaipe_label(label_drugname: Dict, ocr):
    new_ocr = {}
    for key, value in ocr.items():
        labels = []
        image_name = key.split('/')[-1]
        for text in value:
            vaipe_label = find_vaipe_label(label_drugname, text)
            for t in vaipe_label:
                labels.append(t)
        new_ocr[image_name] = labels
    return new_ocr
    
# This is the function which changes form to be easy to process.    
def change_form(pill_pres_map):  
    pres_pill = {}
    
    for dic in pill_pres_map:
        pres_name = dic['pres'].split('.')[0] + '.png'
        pill_names = []
        for pill in dic['pill']:
            pill_name = pill.split('.')[0] + '.jpg'
            pill_names.append(pill_name)
        pres_pill[pres_name] = pill_names
        
    return pres_pill

def map_to_final_result(od_results, ocr_result, pill_pres_map):
    # od_results is result of step 1, ocr_result is result of step 2
    fin_res = {}
    
    pres_pill = change_form(pill_pres_map)
    
    for key, value in ocr_result.items():
        for pill in pres_pill[key]:
            labels = []
            if pill not in od_results:
                print(f'Not found annotations on {pill}')
                continue
            for label in od_results[pill]:
                if label['class_id'] not in value:
                    class_id = 107
                else:
                    class_id = label['class_id']
                labels.append({'class_id': class_id, 'x_min': label['x_min'], 'y_min': label['y_min'], 'x_max': label['x_max'], 'y_max': label['y_max'], 'confidence_score': label['confidence_score']})
            fin_res[pill] = labels
    return fin_res
