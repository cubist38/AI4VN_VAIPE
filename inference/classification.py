from algorithms.classification.test import get_prediction
from algorithms.classification.data_loader.utils import get_test_dataloader
from algorithms.classification.models import arch_map

from typing import Dict
import numpy as np
import pandas as pd
import torch
import os

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
        model = arch_map[backbone](cfg['num_classes'])

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
    model = arch_map[classifier_cfg['backbone']](classifier_cfg['num_classes'])
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