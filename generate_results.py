from typing import Dict
import torch
import os
import yaml
import json
import pandas as pd

from inference.classification import classifier_multi_models
from inference.detection import run_detection
from inference.ocr import run_ocr
from inference.utils import *
from utilities.dir import create_directory

# ==================== MAIN ======================= #

if __name__ == '__main__':
    # Load config file
    cfg = yaml.safe_load(open('configs/config_inference.yaml'))

    create_directory(cfg['detection']['augment_dir'])
    create_directory(cfg['crop']['crop_img_dir'])

    device = torch.device(cfg['device'])

    # ocr -> run detection + augmentation -> crop bbox images -> run classification
    print('Running OCR...')
    # run_ocr(cfg['pres_image_dir'], output_dir=cfg['ocr']['output'])
    print('Compled OCR!')

    print('Running Detection...')
    detection_results = run_detection(cfg['pill_image_dir'], cfg['detection'])
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