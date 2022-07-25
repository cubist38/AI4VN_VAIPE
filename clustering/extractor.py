import torch
from torch import nn
from typing import List, Dict
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import PillDataset

def get_features(cfg: Dict, model: nn.Module, image_paths: List[str]):
    data = PillDataset(image_paths)
    loader = DataLoader(data, batch_size=cfg['batch_size'], shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    predictions = torch.zeros(len(loader.dataset), cfg['feature_dim'])
    idx = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total = len(loader))
        for it, (path, img) in pbar:
            batch_size = img.size(0)
            img = img.cuda()
            pred = model(img)
            predictions[idx:idx + batch_size, :] = pred.squeeze(-1)

            idx += batch_size

    features = predictions.cpu().numpy()
    return features