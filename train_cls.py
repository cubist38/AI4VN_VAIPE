import yaml
import torch
from classification.data_loader.utils import get_dataloader
from classification.models.swin_transformer import swin_tiny_transformer
from classification.trainer import PillTrainer
import json

if __name__ == '__main__':
    cfg = yaml.safe_load(open('configs/config_cls.yaml'))
    with open(cfg['data_dict']) as f:
        data_dict = json.load(f)

    device = torch.device(cfg['device'])
    train_loader, valid_loader = get_dataloader(cfg['img_src'], data_dict)

    model = swin_tiny_transformer(cfg['num_classes'])
    model = model.to(device)

    trainer = PillTrainer(cfg['trainer'], model, device, data_loaders=[train_loader, valid_loader])
    trainer.train()