from typing import Dict
import torch
import os
import yaml
import cv2
import json
import pandas as pd
import numpy as np
import albumentations as A
from ensemble_boxes import *

from ocr.pres_ocr import pres_ocr
from classification.test import get_prediction
from classification.data_loader.utils import get_test_dataloader
from classification.models import swin_transformer_map

from utilities.inference import *


def run_ocr(image_dir: str, output_dir: str) -> Dict:
    ocr_result = pres_ocr(image_dir=image_dir)
    with open(output_dir, 'w', encoding='utf8') as f:
        json.dump(ocr_result, f, ensure_ascii=False)


def run_detection(image_folder: str, augment_folder: str, detection_cfg: Dict, crop_cfg: Dict, model_name: str = 'yolov5') -> Dict:
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
        model.conf = detection_cfg['model_conf']
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
                outputs = model(path)
                df = outputs.pandas().xyxy[0]
                xmins, ymins, xmaxs, ymaxs = df['xmin'], df['ymin'], df['xmax'], df['ymax']
                confs = df['confidence']
                class_ids = df['class']
                bbox = None
                for i in range(len(xmins)):
                    if idx == 1:
                        bbox = revert_from_Rotate180([xmins[i], ymins[i], xmaxs[i], ymaxs[i]], shape[1], shape[0], confs[i], class_ids[i])
                    else:
                        bbox = [xmins[i], ymins[i], xmaxs[i], ymaxs[i], confs[i], class_ids[i]]
                    boxes_list.append([bbox[0], bbox[1], bbox[2], bbox[3]])
                    scores_list.append(bbox[4])
                    labels_list.append(bbox[5])
            
                normalized_bboxes_list = normalized_bbox_coordinate(boxes_list, shape[1], shape[0])
                normalized_bboxes, scores, labels = weighted_boxes_fusion([normalized_bboxes_list], [scores_list], [labels_list], weights=None, iou_thr=0.55, skip_box_thr=0.0)
                bboxes = normalized_to_real(normalized_bboxes, shape[1], shape[0])

            for i in range(len(bboxes)):
                xmin, ymin, xmax, ymax = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
                boxes.append((bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3], labels[i], scores[i]))
                crop_image_name = str(i) + '_' + file    
                crop_img = mycrop(img, xmin, ymin, xmax, ymax, 5)
                crop_detection_map[crop_image_name] = {
                    'image_id': file,
                    'x_min': xmin,
                    'y_min': ymin,
                    'x_max': xmax,
                    'y_max': ymax,
                }
                cv2.imwrite(os.path.join(crop_cfg['crop_img_dir'], crop_image_name), crop_img)
            
            detection_results[file] = boxes    
        with open(crop_cfg['crop_detection_map'], "w") as f:
            json.dump(crop_detection_map, f)
    else:
        print('This model_name is not valid')
        return None
    
    return detection_results

def classifier_multi_models(cfg: Dict, device):
    image_names = os.listdir(cfg['test_img_dir'])
    total_pred_vectors = np.zeros((len(image_names), cfg['num_classes']))
    for id in range(cfg['num_models']):
        print('MODEL:', cfg['backbone'][id])
        print('=' * 40)
        backbone = cfg['backbone'][id]
        cur_cfg = {
            'img_size': cfg['img_size'][id],
            'batch_size': cfg['batch_size'],
        }
        test_loader = get_test_dataloader(cur_cfg, cfg['test_img_dir'])
        model = swin_transformer_map[backbone](cfg['num_classes'])

        pred_vectors = np.zeros((len(image_names), cfg['num_classes']))
        model_list = os.listdir(cfg['weight_path'][id])
        print(f'Found {len(model_list)} models!')
        for model_name in model_list:
            model_path = os.path.join(cfg['weight_path'][id], model_name)
            model.load_state_dict(torch.load(model_path))
            print('Load model ', model_name)
            model = model.to(device)
            _pred_vectors = get_prediction(model, test_loader, device, get_predict_score=True)
            pred_vectors = pred_vectors + _pred_vectors
        
        pred_vectors /= len(model_list)
        total_pred_vectors += cfg['model_weight'][id] * pred_vectors

    predictions = np.argmax(total_pred_vectors, axis=1)
    confidences = np.array([total_pred_vectors[idx, label] for idx, label in enumerate(predictions)])
    df = pd.DataFrame({'image_id': image_names, 'prediction': predictions, 'confidence': confidences})
    df.to_csv(cfg['output'])


# ==================== MAIN ======================= #

if __name__ == '__main__':
    # Load config file
    cfg = yaml.safe_load(open('configs/config_inference.yaml'))

    if not os.path.exists(cfg['augment_dir']):
        os.mkdir(cfg['augment_dir'])

    if not os.path.exists(cfg['crop']):
        os.mkdir(cfg['crop'])

    device = torch.device(cfg['device'])

    # ocr -> run detection + augmentation -> crop bbox images -> run classification
    print('Running OCR...')
    run_ocr(cfg['pres_image_dir'], output_dir=cfg['ocr']['output'])
    print('Compled OCR!')

    print('Running Detection...')
    detection_results = run_detection(cfg['pill_image_dir'], cfg['augment_dir'], cfg['detection'], cfg['crop'])
    print('Compled Detection!')

    print('Running Classification...')
    classifier_multi_models(cfg['classifier_multi_models'], device)
    print('Completed Classification...')

    # generate submission
    print('Generating submit file ...', end = ' ')
    # load all neccessary files
    with open(cfg['ocr']['output'], 'r') as f:
        ocr_output_dict = json.load(f)
    with open(cfg['label_drugnames_path'], 'r') as f:
        label_drugname = json.load(f)
    with open(cfg['pill_pres_map_path'], 'r') as f:
        pill_pres_map = json.load(f)
    with open(cfg['crop']['crop_detection_map'], 'r') as f:
        crop_detection_map = json.load(f)
    
    classifier_df = pd.read_csv(cfg['classifier']['output'])

    ocr_result = text_to_vaipe_label(label_drugname, ocr_output_dict)

    od_results = {}
    for i in range(len(classifier_df)):
        image_id = classifier_df['image_id'][i]
        prediction = classifier_df['prediction'][i]
        confidence = classifier_df['confidence'][i]
        annotation = crop_detection_map[image_id]
        assert prediction < 107
        item = {
            'x_min': annotation['x_min'], 
            'y_min': annotation['y_min'], 
            'x_max': annotation['x_max'], 
            'y_max': annotation['y_max'],
            'class_id': prediction if confidence > cfg['classifier']['threshold'] else 107,
            #'class_id': prediction,
            'confidence_score': confidence
        }
        if annotation['image_id'] not in od_results:
            od_results[annotation['image_id']] = []
        # print(annotation['image_id'])
        od_results[annotation['image_id']].append(item)

    fin_res = map_to_final_result(od_results, ocr_result, pill_pres_map)

    class_id = []
    x_min = []
    x_max = []
    y_min = []
    y_max = []
    confidence_score = []
    image_name = []

    for key, value in fin_res.items():
        for dic in value:
            image_name.append(key)
            class_id.append(int(dic['class_id']))
            confidence_score.append(dic['confidence_score'])
            x_min.append(dic['x_min'])
            y_min.append(dic['y_min'])
            x_max.append(dic['x_max'])
            y_max.append(dic['y_max'])
            
    results_1 = {'image_name': image_name, 'class_id': class_id, 'confidence_score': confidence_score, 'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}
    df = pd.DataFrame(data = results_1)
    df.to_csv(cfg['submit_file'], index = False)
    print('Successfully!')
    print('Submission has been saved at ',  cfg['submit_file'])
