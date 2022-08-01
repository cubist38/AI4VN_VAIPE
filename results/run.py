import os
from detection.run import do_detection
from ocr.pres_ocr import pres_ocr
import pandas as pd
import json
from typing import Dict

def run_ocr(image_dir: str) -> Dict:
    # # Run OCR algorithm
    raw_result = pres_ocr(image_dir=image_dir)

    # # Save for later run
    # with open('./tmp_ocr.json', 'w', encoding='utf8') as f:
        # json.dump(raw_result, f, ensure_ascii=False)
    # # Load from saved result
    # raw_result = json.load(open('./tmp_ocr.json', 'r', encoding='utf8'))

    raw_dict = {}
    for item in raw_result:
        image_path, drugnames = item
        raw_dict[image_path] = []
        for drug in drugnames:
            raw_dict[image_path].append(drug)

    # Maps from text to vaipe's label.
    label_drugnames_path = 'data/label_drugnames.json'
    with open(label_drugnames_path, 'r', encoding='utf-8') as f:
        label_drugnames = json.load(f)
        
    def find_vaipe_label(text):
        for t in text:
            for dic in label_drugnames:
                for d in dic['drugnames']:
                    if text == d:
                        return dic['label']
        return -1
    def text_to_vaipe_label(ocr):
        new_ocr = {}
        for key, value in ocr.items():
            labels = []
            image_name = key.split('/')[-1]
            for text in value:
                vaipe_label = find_vaipe_label(text)
                labels.append(vaipe_label)
            new_ocr[image_name] = labels
        return new_ocr
        
    ocr_result = text_to_vaipe_label(raw_dict)
    return ocr_result


def run_od(image_dir: str) -> Dict:
    # Run OD algorithm
    results = do_detection(image_dir, model_name='yolov5')

    # Get the real label
    with open('data/label_freq.json') as f:
        label_freq = json.load(f)
    
    def get_vaipe_label(kmeans_label):
        freq_kmeans_label = label_freq[str(kmeans_label)][:107]
        vaipe_label = freq_kmeans_label.index(max(freq_kmeans_label))
        return vaipe_label
                 
    def kmeans_to_vaipe(r):
        new_r = {}
        for key, value in r.items():
            annotation = []
            tmp = key.replace('\\', '/')
            image_name = tmp.split('/')[-1]
            for tup in value:
                x_min, y_min, x_max, y_max, class_id, confidence_score = tup
                vaipe_class_id = get_vaipe_label(class_id)
                annotation.append({'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max, 'class_id': vaipe_class_id, 'confidence_score': confidence_score})
                
            new_r[image_name] = annotation
            
        return new_r
            
    new_results = kmeans_to_vaipe(results)
    return new_results

def find_out_of_pres(od_result: Dict, ocr_result: Dict, pill_pres_map_path: str) -> Dict:
    with open(pill_pres_map_path, 'r') as f:
        pill_pres_map = json.load(f)
        
    # Changes form to be easy to process.    
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

    def map_to_final_result(od_result, ocr_result, pill_pres_map):
        fin_res = {}
        pres_pill = change_form(pill_pres_map)
        for key, value in ocr_result.items():
            for pill in pres_pill[key]:
                labels = []
                for label in od_result[pill]:
                    if label['class_id'] not in value:
                        class_id = 107
                    else:
                        class_id = label['class_id']
                    labels.append({'class_id': class_id, 'x_min': label['x_min'], 'y_min': label['y_min'], 'x_max': label['x_max'], 'y_max': label['y_max'], 'confidence_score': label['confidence_score']})
                fin_res[pill] = labels
        return fin_res
    
    final_result = map_to_final_result(od_result, ocr_result, pill_pres_map)
    return final_result


def get_result(data_path: str, output_path: str = './results/csv/result.csv'):
    '''
        Generate result file for data in `data_path`
        The structure of folder at `data_path` should be in the following format:
        ---data_path/
            |---pill/
                |---image/
            |---prescription/
                |---image/
            |---pill_pres_map.json
    '''
    # 1. OCR for prescription
    pres_folder = os.path.join(data_path, 'prescription/image')
    ocr_results = run_ocr(pres_folder)
    print('Compelete OCR steps...')

    # 2. Object detection for pill
    pills_folder = os.path.join(data_path, 'pill/image')
    od_results = run_od(pills_folder)
    print('Compelete Object detection steps...')

    # 3. Final result
    map_path = os.path.join(data_path, 'pill_pres_map.json')
    final_results = find_out_of_pres(od_results, ocr_results, map_path)
    print('Compelete Final steps...')

    # To CSV
    class_id, x_min, x_max, y_min, y_max = [], [], [], [], []
    confidence_score = []
    image_name = []

    for key, value in final_results.items():
        for dic in value:
            image_name.append(key)
            class_id.append(dic['class_id'])
            confidence_score.append(dic['confidence_score'])
            x_min.append(dic['x_min'])
            y_min.append(dic['y_min'])
            x_max.append(dic['x_max'])
            y_max.append(dic['y_max'])
            
    tmp = {'image_name': image_name, 'class_id': class_id, 'confidence_score': confidence_score, 'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}
    df = pd.DataFrame(data = tmp)
    df.to_csv(output_path, index = False)