from torch import nn
import torch
from tqdm import tqdm
import numpy as np

def get_prediction(model: nn.Module, loader, device, threshold: float = 0.5):
    model.eval()
    idx = 0
    predictions = torch.zeros(len(loader.dataset), 1)
    confidences = torch.zeros(len(loader.dataset), 1)
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total = len(loader))
        for it, img in pbar:
            last_it = it
            batch_size = img.size(0)
            img = img.to(device)

            pred = model(img)
            pred_label = torch.argmax(pred, 1).unsqueeze(-1)
            # print(torch.amax(pred, 1).unsqueeze(-1).shape)
            predictions[idx:idx + batch_size, :] = pred_label
            confidences[idx:idx + batch_size, :] = torch.amax(pred, 1).unsqueeze(-1)

            idx += batch_size
            # pbar.set_description(description)

    predictions = np.squeeze(predictions.cpu().numpy())
    confidences = np.squeeze(confidences.cpu().numpy())
    return predictions, confidences
