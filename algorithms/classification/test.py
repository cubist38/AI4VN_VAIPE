from torch import nn
import torch
from tqdm import tqdm
import numpy as np

def get_eval_prediction(model: nn.Module, loader, device):
    model.eval()
    idx = 0
    predictions = torch.zeros(len(loader.dataset), 1)
    confidences = torch.zeros(len(loader.dataset), 1)
    labels = torch.zeros(len(loader.dataset), 1)
    pred_list = []
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total = len(loader))
        for it, (img, label) in pbar:
            last_it = it
            batch_size = img.size(0)
            img = img.to(device)

            pred = model(img)
            pred_list.append(pred)
            pred_label = torch.argmax(pred, 1).unsqueeze(-1)
            predictions[idx:idx + batch_size, :] = pred_label
            labels[idx:idx + batch_size, :] = torch.argmax(label, 1).unsqueeze(-1)
            confidences[idx:idx + batch_size, :] = torch.amax(pred, 1).unsqueeze(-1)

            idx += batch_size
            # pbar.set_description(description)
    predictions = np.squeeze(predictions.cpu().numpy())
    labels = np.squeeze(labels.cpu().numpy())
    confidences = np.squeeze(confidences.cpu().numpy())
    return predictions, labels, confidences

def get_prediction(model: nn.Module, loader, device, get_predict_score: bool = False):
    model.eval()
    idx = 0
    predictions = torch.zeros(len(loader.dataset), 1)
    confidences = torch.zeros(len(loader.dataset), 1)
    pred_list = []
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total = len(loader))
        for it, img in pbar:
            last_it = it
            batch_size = img.size(0)
            img = img.to(device)

            pred = model(img)
            pred_list.append(pred)
            pred_label = torch.argmax(pred, 1).unsqueeze(-1)
            # print(torch.amax(pred, 1).unsqueeze(-1).shape)
            predictions[idx:idx + batch_size, :] = pred_label
            confidences[idx:idx + batch_size, :] = torch.amax(pred, 1).unsqueeze(-1)

            idx += batch_size
            # pbar.set_description(description)

    pred_vectors = torch.cat(pred_list, dim = 0)
    pred_vectors = pred_vectors.cpu().numpy()
    predictions = np.squeeze(predictions.cpu().numpy())
    confidences = np.squeeze(confidences.cpu().numpy())
    if get_predict_score:
        return pred_vectors
    return predictions, confidences
