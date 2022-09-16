from typing import Dict
import shutil
import cv2
import json
import os
import albumentations as A
from ensemble_boxes import *
from utilities.dir import create_directory
import numpy as np

# ====================== Detection ==================== #

def convert_to_original_shape(boxes, old_w, old_h, w, h, target_size = 640):
    
    new_boxes = []

    x_scale = old_w / w
    y_scale = old_h / h
    
    for bbox in boxes:
        xmin, ymin, xmax, ymax, class_id, confidence_score = bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max'], bbox['class_id'], bbox['confidence_score']
        new_xmin = int(np.round(xmin * x_scale))
        new_ymin = int(np.round(ymin * y_scale))
        new_xmax = int(np.round(xmax * x_scale))
        new_ymax = int(np.round(ymax * y_scale))
        new_boxes.append({'x_min': new_x, 'y_min': new_y, '': new_w, 'h': new_h, 'class_id': class_id, 'confidence_score': confidence_score})

    return new_boxes
    
    

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

def mycropoffset(img, x_min, y_min, x_max, y_max, offset = 10):
    if x_min - offset > 0:
        x_min = x_min - offset
    if y_min - offset > 0:
        y_min = y_min - offset
    
    x_max = x_max + offset
    y_max = y_max + offset
    
    return img[y_min : y_max, x_min : x_max]

def check_iou(bbox1, bbox2, iou_thr):
    x_min = max(bbox1[0], bbox2[0])
    y_min = max(bbox1[1], bbox2[1])
    x_max = min(bbox1[2], bbox2[2])
    y_max = min(bbox1[3], bbox2[3])
    
    interArea = max(x_max - x_min, 0) * max(y_max - y_min, 0)
    
    bbox1Area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2Area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    iou = interArea / (bbox1Area + bbox2Area - interArea)
    
    if iou > iou_thr:
        return True
    return False
    
# ====================== MAPPING ==================== #

# This is the function which maps from text to vaipe's label.
def rename(label_drugname: Dict, text):
    if text in label_drugname:
        return text
    tokenizer = text.split(' ')
    mass = tokenizer[-1]
    if mass[-2:] == "mg":
        m = float(mass[:-2])/1000
        new_mass = (str(m) + 'g').replace('.', ',')
    elif mass[-1:] == 'g':
        m = int(float(mass[:-1].replace(',', '.')) * 1000)
        new_mass = (str(m) + 'mg')
    else:
        return None
    new_text = ""
    for i in range(len(tokenizer) - 1):
        new_text += tokenizer[i] + ' '
    new_text += new_mass
    return new_text

def find_vaipe_label(label_drugname: Dict, text):
    text = rename(label_drugname, text)
    if text:
        return label_drugname[text]
    return None

# We should pass the ocr_output_dict to this function to have the mapping.
def text_to_vaipe_label(label_drugname: Dict, ocr):
    new_ocr = {}
    for key, value in ocr.items():
        labels = []
        image_name = key.split('/')[-1]
        for text in value:
            vaipe_label = find_vaipe_label(label_drugname, text)
            if vaipe_label:
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
