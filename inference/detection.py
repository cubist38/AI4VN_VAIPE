from typing import Dict
import torch
import os
import cv2

def run_detection(image_folder: str, detection_cfg: Dict, model_name: str = 'yolov5') -> Dict:
    '''
        Run Object Detection model to return the result of images in image_folder
        Returns:
            A dictonary with:
            - key: path_to_image
            - value: a list of objects with the following information: (xmin, ymin, xmax, ymax, label)
    '''
    detection_results = {}

    if model_name == 'yolov5':
        model = torch.hub.load('detection/yolo/yolov5', 'custom', path=detection_cfg['weight_path'], source='local')
        image_files = os.listdir(image_folder)
        for id, file in enumerate(image_files):
            print(f'[{id+1}/{len(image_files)}] Running on {file} ...')
            path = os.path.join(image_folder, file)
            img = cv2.imread(path)
            H, W = img.shape[:2]
            model.conf = detection_cfg['model_conf']
            outputs = model(path)
            df = outputs.pandas().xyxy[0]
            xmins, ymins, xmaxs, ymaxs = df['xmin'], df['ymin'], df['xmax'], df['ymax']
            confs = df['confidence']
            labels = df['class']
            boxes = []
            for i in range(len(xmins)):
                xmin, ymin, xmax, ymax = int(xmins[i]), int(ymins[i]), int(xmaxs[i]), int(ymaxs[i])
                conf, label = confs[i], labels[i]
                border_h = int((ymax-ymin) * detection_cfg['bbox_extend_percent'])
                border_w = int((xmax-xmin) * detection_cfg['bbox_extend_percent'])
                xmin = max(0, xmin - border_w)
                ymin = max(0, ymin - border_h)
                xmax = min(W-1, xmax + border_w)
                ymax = min(H-1, ymax + border_h)
                boxes.append((xmin, ymin, xmax, ymax, label, conf))
            detection_results[path] = boxes
    else:
        print('This model_name is not valid')
        return None
    
    return detection_results