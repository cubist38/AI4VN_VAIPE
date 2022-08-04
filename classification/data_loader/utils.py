import os
from typing import Dict
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import PillDataset, TestPillDataset
from sklearn.model_selection import train_test_split

def get_dataloader(cfg: Dict, data_dict: Dict):
    image_paths, image_labels = [], []
    for img_name, label in data_dict.items():
        if label != 107:
            image_paths.append(os.path.join(cfg['img_src'], str(label), img_name))
            image_labels.append(label)
    
    print(f'Found {len(image_labels)} images has label != 107')

    X_train, X_test, y_train, y_test = train_test_split(image_paths, image_labels, 
                                                        test_size=0.3, random_state=2022)

    print(f'N.Train = {len(X_train)}, N.Test = {len(X_test)}')

    train_dataset = PillDataset(X_train, y_train, cfg['img_size'], get_train_transformer(), cfg['num_classes'])
    valid_dataset = PillDataset(X_test, y_test, cfg['img_size'], get_valid_transformer(), cfg['num_classes']) 

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg['batch_size']*2, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return train_loader, valid_loader

def get_test_dataloader(cfg: Dict, test_img_dir: str):
    image_paths = [os.path.join(test_img_dir, img_name) for img_name in os.listdir(test_img_dir)]
    print(f'Found {len(image_paths)} in {test_img_dir}')
    test_dataset = TestPillDataset(image_paths, cfg['img_size'], get_valid_transformer())
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size']*2, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    
    return test_loader

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