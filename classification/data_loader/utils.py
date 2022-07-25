import numpy as np
import os
from typing import Dict
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import PillDataset
from sklearn.model_selection import train_test_split

def get_dataloader(cfg: Dict, data_dict: Dict):
    image_paths, image_labels = [], []
    for img_name, label in data_dict.items():
        if label != 107:
            image_paths.append(os.path.join(cfg['img_src'], img_name))
            image_labels.append(label)
    
    print(f'Found {len(image_labels)} images has label != 107')

    X_train, X_test, y_train, y_test = train_test_split(image_paths, image_labels, 
                                                        test_size=0.3, random_state=2022)

    print(f'N.Train = {len(X_train)}, N.Test = {len(X_test)}')

    train_dataset = PillDataset(X_train, y_train, cfg['img_size'], get_train_transformer())
    valid_dataset = PillDataset(X_test, y_test, cfg['img_size'], get_valid_transformer()) 

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return train_loader, valid_loader

def get_train_transformer():
    transformer = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transformer


def get_valid_transformer():
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transformer