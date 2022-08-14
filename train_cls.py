import yaml
import torch
import json
import os
import random
from typing import Dict
from sklearn.model_selection import train_test_split, StratifiedKFold

from utilities.seed import seed_torch
from classification.data_loader.utils import get_dataloader
from classification.models import swin_transformer_map
from classification.trainer import PillTrainer

def train_single_fold(fold_id: int, X_train, X_test, y_train, y_test, model, device, cfg: Dict):
    train_loader, valid_loader = get_dataloader(cfg, X_train, X_test, y_train, y_test)
    trainer = PillTrainer(cfg['trainer'], model, device, data_loaders=[train_loader, valid_loader], fold_id = fold_id)
    trainer.train()

def remove_sample(image_paths: str, image_labels: str):
    label_cnt = [[] for _ in range(107)]
    for id, label in enumerate(image_labels):
        label_cnt[label].append(id)
    
    sample_id = []
    for i in range(107):
        if len(label_cnt[i]) > 200:
            label_cnt[i] = random.sample(label_cnt[i], 200)
        sample_id += label_cnt[i]
    
    image_paths = [image_paths[id] for id in sample_id]
    image_labels = [image_labels[id] for id in sample_id]
    return image_paths, image_labels

def get_weighted_loss(image_labels):
    label_cnt = [0 for _ in range(107)]
    for label in image_labels:
        label_cnt[label] += 1
    loss_weights = [0 for _ in range(107)]
    for idx in range(107):
        loss_weights[idx] = min(label_cnt)/label_cnt[idx]
    print('Loss weights:')
    print(loss_weights)
    return loss_weights

if __name__ == '__main__':
    seed_torch(2022)
    cfg = yaml.safe_load(open('configs/config_cls.yaml'))
    with open(cfg['data_dict']) as f:
        data_dict = json.load(f)
    device = torch.device(cfg['device'])

    image_paths, image_labels = [], []
    for item in data_dict:
        if item['label'] != 107:
            image_paths.append(os.path.join(cfg['img_src'], str(item['label']), item['image']))
            image_labels.append(item['label'])
    print(f'Found {len(image_labels)} images has label != 107')
    image_paths, image_labels = remove_sample(image_paths, image_labels)

    cfg['trainer']['loss_weight'] = get_weighted_loss(image_labels)

    if cfg['k_fold'] == 0:
        print('Not using k-fold')
        print('Model:', cfg['trainer']['backbone'])
        model = swin_transformer_map[cfg['trainer']['backbone']](cfg['num_classes'])
        model = model.to(device)
        X_train, X_test, y_train, y_test = train_test_split(image_paths, image_labels, 
                                                        test_size=0.3, random_state=2022)
        train_single_fold(None, X_train, X_test, y_train, y_test, model, device, cfg)
    else:
        num_fold = cfg['k_fold']
        print(f'Using {num_fold}-fold')
        kf = StratifiedKFold(n_splits=cfg['k_fold'], shuffle=True, random_state=2022)
        for fold_id, (train_index, test_index) in enumerate(kf.split(image_paths, image_labels)):
            X_train, X_test = [image_paths[id] for id in train_index], [image_paths[id] for id in test_index]
            y_train, y_test = [image_labels[id] for id in train_index], [image_labels[id] for id in test_index]
            print('=' * 30)
            print(f'TRAINING FOLD {fold_id}')

            model = swin_transformer_map[cfg['trainer']['backbone']](cfg['num_classes'])
            model = model.to(device)
            train_single_fold(fold_id, X_train, X_test, y_train, y_test, model, device, cfg)
            del model