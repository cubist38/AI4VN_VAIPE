from typing import Dict
import torch
import os
import cv2
from inference.utils import *

def run_detection(image_folder: str, augment_folder: str, detection_cfg: Dict, crop_cfg: Dict, model_name: str = 'yolov5') -> Dict:
    '''
        Run Object Detection model to return the result of images in image_folder
        Returns:
            A dictonary with:
            - key: path_to_image
            - value: a list of objects with the following information: (xmin, ymin, xmax, ymax, label)
    '''
    detection_results = {}

    if not os.path.exists(augment_folder):
        os.mkdir(augment_folder)
    if not os.path.exists(crop_cfg['crop_img_dir']):
        os.mkdir(crop_cfg['crop_img_dir'])

    if model_name == 'yolov5':
        model = torch.hub.load('detection/yolo/yolov5', 'custom', path=detection_cfg['weight_bbox_only_path'], source='local')
        model1 = torch.hub.load('detection/yolo/yolov5', 'custom', path=detection_cfg['weight_bbox_label_path'], source='local')
        model.conf = detection_cfg['model_conf']
        model1.conf = detection_cfg['model_conf']
        image_files = os.listdir(image_folder)
        crop_detection_map = {}
        
        for id, file in enumerate(image_files):
            print(f'[{id+1}/{len(image_files)}] Running on {file} ...')
            img_path = os.path.join(image_folder, file)
            path = None
            img = cv2.imread(img_path)
            augment_image(img, file, augment_folder)
            shape = img.shape
            
            boxes = []
            boxes_list = []
            scores_list = []
            labels_list = []
            bboxes = []
            
            
            for idx in range(4):
                if idx == 0:
                    path = img_path
                else:
                    path = os.path.join(augment_folder, file.split('.')[0] + '_' + str(idx) + '.jpg')
                outputs = model(path, size = 640)
                df = outputs.pandas().xyxy[0]
                xmins, ymins, xmaxs, ymaxs = df['xmin'], df['ymin'], df['xmax'], df['ymax']
                confs = df['confidence']
                class_ids = df['class']
                bbox = None
                for i in range(len(xmins)):
                    if idx == 1:
                        bbox = revert_from_Rotate180([xmins[i], ymins[i], xmaxs[i], ymaxs[i]], shape[1], shape[0], confs[i], 107)
                    else:
                        bbox = [xmins[i], ymins[i], xmaxs[i], ymaxs[i], confs[i], 107]
                        
                    boxes_list.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                    scores_list.append(bbox[4])
                    labels_list.append(bbox[5])
            
                normalized_bboxes_list = normalized_bbox_coordinate(boxes_list, shape[1], shape[0])
                normalized_bboxes, scores, labels = weighted_boxes_fusion([normalized_bboxes_list], [scores_list], [labels_list], weights=None, iou_thr=0.55, skip_box_thr=0.0)
                bboxes = normalized_to_real(normalized_bboxes, shape[1], shape[0])
            
            outputs = model1(img_path, size = 640)
            df = outputs.pandas().xyxy[0]
            xmins, ymins, xmaxs, ymaxs = df['xmin'], df['ymin'], df['xmax'], df['ymax']
            confs = df['confidence']
            class_ids = df['class']
            label_boxes = []
            
            
            for i in range(len(xmins)):
                label_boxes.append([xmins[i], ymins[i], xmaxs[i], ymaxs[i], confs[i], class_ids[i]])
            
            fin_boxes_list = []
            fin_labels_list = []
            fin_scores_list = []
            
            for i in range(len(bboxes)):
                xmin, ymin, xmax, ymax = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
                fin_boxes_list.append([xmin, ymin, xmax, ymax])
                fin_labels_list.append(labels[i])
                fin_scores_list.append(scores[i])
                
            for i in range(len(fin_boxes_list)):
                bbox1 = fin_boxes_list[i]
                for label_bbox in label_boxes:
                    bbox2 = [label_bbox[0], label_bbox[1], label_bbox[2], label_bbox[3]]
                    if check_iou(bbox1, bbox2, 0.6) == True:
                        fin_labels_list[i] = label_bbox[5]
                        fin_scores_list[i] = (label_bbox[4] + fin_scores_list[i])/2
                        
            for label_bbox in label_boxes:
                fin_boxes_list.append([label_bbox[0], label_bbox[1], label_bbox[2], label_bbox[3]])
                fin_labels_list.append(label_bbox[5])
                fin_scores_list.append(label_bbox[4])
            
            fin_normalized_bboxes_list = normalized_bbox_coordinate(fin_boxes_list, shape[1], shape[0])
            fin_normalized_bboxes, fin_scores, fin_labels = weighted_boxes_fusion([fin_normalized_bboxes_list], [fin_scores_list], [fin_labels_list], weights=None, iou_thr=0.55, skip_box_thr=0.0)
            fin_bboxes = normalized_to_real(fin_normalized_bboxes, shape[1], shape[0])
            
            boxes = []
                
            for i in range(len(fin_bboxes)):
                xmin, ymin, xmax, ymax = int(fin_bboxes[i][0]), int(fin_bboxes[i][1]), int(fin_bboxes[i][2]), int(fin_bboxes[i][3])
                if int(fin_labels[i]) == 107:
                    crop_image_name = str(i) + '_' + file    
                    crop_img = img[ymin : ymax, xmin : xmax]
                    crop_img = mycropoffset(img, xmin, ymin, xmax, ymax, 5)
                    crop_detection_map[crop_image_name] = {
                        'image_id': file,
                        'x_min': xmin,
                        'y_min': ymin,
                        'x_max': xmax,
                        'y_max': ymax,
                    }
                    cv2.imwrite(os.path.join(crop_cfg['crop_img_dir'], crop_image_name), crop_img)
                else:
                    boxes.append({'x_min': xmin, 'y_min': ymin, 'x_max': xmax, 'y_max': ymax, 'class_id': int(fin_labels[i]), 'confidence_score': fin_scores[i]})
                
                
            detection_results[file] = boxes  
        with open(crop_cfg['crop_detection_map'], "w") as f:
            json.dump(crop_detection_map, f)
    else:
        print('This model_name is not valid')
        return None
    
    return detection_results
