from typing import Dict
import torch
from torch import optim, nn
from tqdm import tqdm
from utils.dir import create_directory
import os

class PillTrainer:
    def __init__(self, cfg: Dict, model, device, data_loaders = None) -> None:
        self.cfg = cfg
        self.device = device
        self.model = model

        if cfg['freeze']:
            print('Applying freeze model header!')
            self.model.net.freeze_header()

        self.num_epochs = cfg['epochs']

        param_groups = [
            {'params': self.model.backbone.parameters()},
            {'params': self.model.classify.parameters(), 'lr': cfg['cls_lr']}
        ]

        self.optimizer = optim.AdamW(param_groups, lr = cfg['lr'])
        # self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0 = 20, eta_min=1e-4)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                        milestones=cfg['optim']['milestone'],
                                                        gamma=cfg['optim']['gamma'])

        if data_loaders is not None:
            [self.train_loader, self.val_loader] = data_loaders
        
        self.criterion = nn.CrossEntropyLoss()
        self.best = 0

    def validate(self, loader, step: int, log = 'valid'):
        print('VALIDATE:')
        self.model.eval()

        total_loss = 0
        idx = 0

        predictions = torch.zeros(len(loader.dataset), 1)
        true_labels = torch.zeros(len(loader.dataset), 1)

        with torch.no_grad():
            pbar = tqdm(enumerate(loader), total = len(loader))
            for it, (img, label) in pbar:
                last_it = it
                batch_size = img.size(0)
                img, label = img.to(self.device), label.to(self.device)

                pred = self.model(img)
                loss = self.criterion(pred, label)
                total_loss += loss.item()
                predictions[idx:idx + batch_size, :] = torch.argmax(pred, 1).unsqueeze(-1)
                true_labels[idx:idx + batch_size, :] = label.unsqueeze(-1)

                idx += batch_size

                avg_loss = total_loss/(it+1)
                description = f"iter: {it}    \t| avg_loss: {avg_loss:.5f}"
                # pbar.set_description(description)

        predictions = predictions.cpu().numpy()
        true_labels = true_labels.cpu().numpy()
        acc = (predictions == true_labels).sum() / len(predictions)
        if acc > self.best:
            self.save_model(self.cfg['save_path'])
            self.best = acc
        print(f'Acc = {acc} | Best = {self.best}')
    
    def train(self):
        step, log_step = 0, 0
        for epoch in range(self.num_epochs):
            self.model.train()
            if self.cfg['freeze'] and epoch == self.cfg['unfreeze_epoch']:
                print('Unfreeze model header!')
                self.model.net.unfreeze_header()
            print(f'[*] Epoch: {epoch}')
            total_loss, avg_loss = 0, 0
            # pbar = tqdm(enumerate(self.train_loader), total = len(self.train_loader))
            for it, (img, label) in enumerate(self.train_loader):
                img, label = img.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()

                pred = self.model(img)

                loss = self.criterion(pred, label)

                total_loss += loss.item()
                avg_loss = total_loss/(it+1)

                description = f"iter: {it}    \t| avg_loss: {avg_loss:.5f}"
                if(it % self.cfg['log_step'] == 0):
                    print(description)
                # pbar.set_description(description)

                loss.backward()
                self.optimizer.step()

                step += 1
                if step % self.cfg['eval_step'] == 0 or it == len(self.train_loader) - 1:
                    self.validate(self.val_loader, log_step)
                    log_step += 1

            self.scheduler.step()

    def save_model(self, path: str):
        create_directory(path)
        torch.save(self.model.state_dict(), os.path.join(path, '{}.pth'.format(self.cfg['model_name'])))
        print('Saved cls model!')
