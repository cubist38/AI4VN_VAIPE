from typing import Dict
import torch
import os
import cv2
from inference.utils import *

def run_detection(image_folder: str, detection_cfg: Dict,  model_name: str = 'yolov5') -> Dict:
    '''
        Run Object Detection model to return the result of images in image_folder
        Returns:
            A dictonary with:
            - key: path_to_image
            - value: a list of objects with the following information: (xmin, ymin, xmax, ymax, label)
    '''
    detection_results = {}

    if model_name == 'yolov5':
        model = torch.hub.load('algorithms/detection/yolo/yolov5', 'custom', path=detection_cfg['weight_path'], source='local')
        model.conf = detection_cfg['model_conf']
        image_files = os.listdir(image_folder)
        
        for id, file in enumerate(image_files):
            print(f'[{id+1}/{len(image_files)}] Running on {file} ...')
            img_path = os.path.join(image_folder, file)
            path = None
            img = cv2.imread(img_path)
            augment_image(img, file, detection_cfg['augment_dir'])
            H, W = img.shape[:2]
            
            boxes_list = []
            scores_list = []
            labels_list = []
            
            for idx in range(4):
                if idx == 0:
                    path = img_path
                else:
                    path = os.path.join(detection_cfg['augment_dir'], file.split('.')[0] + '_' + str(idx) + '.jpg')
                outputs = model(path, size = 640)
                df = outputs.pandas().xyxy[0]
                xmins, ymins, xmaxs, ymaxs = df['xmin'], df['ymin'], df['xmax'], df['ymax']
                confs = df['confidence']
                class_ids = df['class']
                bbox = None
                for i in range(len(xmins)):
                    if idx == 1:
                        bbox = revert_from_Rotate180([xmins[i], ymins[i], xmaxs[i], ymaxs[i]], W, H, confs[i], class_ids[i])
                    else:
                        bbox = [xmins[i], ymins[i], xmaxs[i], ymaxs[i], confs[i], class_ids[i]]
                    boxes_list.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                    scores_list.append(bbox[4])
                    labels_list.append(bbox[5])
            
                normalized_bboxes_list = normalized_bbox_coordinate(boxes_list, W, H)
                normalized_bboxes, scores, labels = weighted_boxes_fusion([normalized_bboxes_list], [scores_list], [labels_list], weights=None, iou_thr=0.55, skip_box_thr=0.0)
                bboxes = normalized_to_real(normalized_bboxes, W, H, detection_cfg['bbox_extend_percent'])

            # saving memory
            for i in range(1, 4):
                os.remove(os.path.join(detection_cfg['augment_dir'], file.split('.')[0] + f'_{i}.jpg'))

            detection_results[img_path] = bboxes    
    else:
        print('This model_name is not valid')
        return None
    
    return detection_results