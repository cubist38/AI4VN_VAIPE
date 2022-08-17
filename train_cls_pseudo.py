import yaml
import torch
import json
import os
from typing import Dict

from utilities.seed import seed_torch
from classification.data_loader.utils import get_dataloader, get_test_dataloader
from classification.models import swin_transformer_map
from classification.trainer import PillTrainer
from classification.test import get_prediction

def train_single_fold(fold_id: int, X_train, X_test, y_train, y_test, model, device, cfg: Dict):
    train_loader, valid_loader = get_dataloader(cfg, X_train, X_test, y_train, y_test)
    trainer = PillTrainer(cfg['trainer'], model, device, data_loaders=[train_loader, valid_loader], fold_id = fold_id)
    trainer.train()

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

    
    # pseudo label
    print('Run pseudo labelling')
    unlabel_image_paths = [os.path.join(cfg['unlabeled_img_src'], name) for name in os.listdir(cfg['unlabeled_img_src'])]
    test_loader = get_test_dataloader(cfg, cfg['unlabeled_img_src'])
    
    model = swin_transformer_map[cfg['trainer']['backbone']](cfg['num_classes'])
    trained_model_path = os.path.join(cfg['trainer']['save_path'], '{}.pth'.format(cfg['trainer']['model_name']))
    model.load_state_dict(torch.load(trained_model_path))
    print('Load model ', trained_model_path)
    model = model.to(device)
    pseudo_label, pseudo_confidence = get_prediction(model, test_loader, device)

    high_confidence_indicies = []
    for idx in range(len(pseudo_confidence)):
        if pseudo_confidence[idx] >= 0.98:
            high_confidence_indicies.append(idx)
    unlabel_image_paths = [unlabel_image_paths[idx] for idx in high_confidence_indicies]
    pseudo_label = pseudo_label[high_confidence_indicies]
    print(f'Found {len(unlabel_image_paths)} images with high confidence')

    # re-train model
    print('Re-train model')
    image_paths += unlabel_image_paths
    image_labels += pseudo_label.astype(int).tolist()

    cfg['trainer']['loss_weight'] = get_weighted_loss(image_labels)

    cfg['trainer']['epochs'] = 3
    cfg['trainer']['lr'] = 0.00005
    cfg['trainer']['cls_lr'] = 0.00005

    train_loader, valid_loader = get_dataloader(cfg, image_paths, image_paths[:10], image_labels, image_labels[:10])
    trainer = PillTrainer(cfg['trainer'], model, device, data_loaders=[train_loader, valid_loader])
    trainer.train()
    trainer.save_model('weights/cls/pseudo_swin_t')