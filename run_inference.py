from typing import Dict
import torch
import os
import yaml
import cv2
import json
import shutil
import pandas as pd
import numpy as np

# from ocr.pres_ocr import pres_ocr
from classification.test import get_prediction
from classification.data_loader.utils import get_test_dataloader
from classification.models import swin_transformer_map
from utilities.dir import create_directory

def run_ocr(image_dir: str, output_dir: str) -> Dict:
    # ocr_result = pres_ocr(image_dir=image_dir)
    with open(output_dir, 'w', encoding='utf8') as f:
        json.dump(ocr_result, f, ensure_ascii=False)

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
                boxes.append((xmin, ymin, xmax, ymax, label, conf))
            detection_results[path] = boxes
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

def classifier(classifier_cfg: Dict, device):
    test_loader = get_test_dataloader(classifier_cfg, classifier_cfg['test_img_dir'])
    image_names = os.listdir(classifier_cfg['test_img_dir'])

    predictions, confidences = np.zeros(len(image_names)), np.zeros(len(image_names))
    model = swin_transformer_map[classifier_cfg['backbone']](classifier_cfg['num_classes'])
    if not classifier_cfg['ensemble']:
        model.load_state_dict(torch.load(classifier_cfg['weight_path']))
        print('Load model ', classifier_cfg['weight_path'])
        model = model.to(device)
        predictions, confidences = get_prediction(model, test_loader, device)
    else:
        pred_vectors = np.zeros((len(image_names), classifier_cfg['num_classes']))
        model_list = os.listdir(classifier_cfg['weight_ensemble_path'])
        print(f'Found {len(model_list)} models!')
        for model_name in model_list:
            model_path = os.path.join(classifier_cfg['weight_ensemble_path'], model_name)
            model.load_state_dict(torch.load(model_path))
            print('Load model ', model_name)
            model = model.to(device)
            _pred_vectors = get_prediction(model, test_loader, device, get_predict_score=True)
            pred_vectors = pred_vectors + _pred_vectors
        
        pred_vectors /= len(model_list)
        predictions = np.argmax(pred_vectors, axis=1)
        confidences = np.array([pred_vectors[idx, label] for idx, label in enumerate(predictions)])

    df = pd.DataFrame({'image_id': image_names, 'prediction': predictions, 'confidence': confidences})
    df.to_csv(classifier_cfg['output'])

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

# ====================== UTILS ==================== #
# This is the function which maps from text to vaipe's label.
def find_vaipe_label(label_drugname: Dict, text):
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


# ==================== MAIN =======================
if __name__ == '__main__':
    # Load config file
    cfg = yaml.safe_load(open('configs/config_inference.yaml'))
    device = torch.device(cfg['device'])

    # ocr -> run detection -> crop bbox images -> run classification
    # Uncommend below line to run OCR (if neccesary)
    # run_ocr(cfg['pres_image_dir'], output_dir=cfg['ocr']['output'])
    # detection_results = run_detection(cfg['pill_image_dir'], cfg['detection'])
    # crop_bbox_images(detection_results, cfg['crop'])
    if cfg['multi_models']:
        classifier_multi_models(cfg['classifier_multi_models'], device)
    else:
        classifier(cfg['classifier'], device)

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
        if image_id not in crop_detection_map:
            print('WTF Not found any annotations in', image_id)
            continue
        annotation = crop_detection_map[image_id]
        assert prediction < 107
        item = {
            'x_min': annotation['x_min'], 
            'y_min': annotation['y_min'], 
            'x_max': annotation['x_max'], 
            'y_max': annotation['y_max'],
            'class_id': prediction if confidence > cfg['classifier']['threshold'] else 107,
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