import torch.utils.data as data
import torch.nn.functional as F
from typing import List
import cv2
import torch
from PIL import Image
import numpy as np
import random
from ..utils import resize_image

class PillDataset(data.Dataset):
    def __init__(self, X: List, y: List, img_size: int, transform, num_classes: int):
        self.image_paths = X
        self.image_labels = y
        self.one_hot_labels = F.one_hot(torch.tensor(self.image_labels), num_classes).to(torch.float32)
        self.img_size = img_size
        self.__image_transformer = transform

    def random_zoom_image(self, image):
        H, W = image.shape[:2]
        h_zoom_p = random.uniform(-0.1, 0.2)
        w_zoom_p = random.uniform(-0.1, 0.2)
        nH, nW = int(H * h_zoom_p), int(W * w_zoom_p)
        
        if h_zoom_p < 0:
            image = image[-nH: H + nH, :]
        else:
            image = cv2.copyMakeBorder(image, nH, nH, 0, 0, cv2.BORDER_REPLICATE)
            
        if w_zoom_p < 0:
            image = image[:, -nW: W + nW]
        else:
            image = cv2.copyMakeBorder(image, 0, 0, nW, nW, cv2.BORDER_REPLICATE)

        return image


    def read_image(self, index: int):
        image = cv2.imread(self.image_paths[index], cv2.COLOR_BGR2RGB)
        image = self.random_zoom_image(image)
        image = resize_image(image, self.img_size)
        return self.__image_transformer(Image.fromarray(image.astype(np.uint8)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img = self.read_image(index)
        return img, self.one_hot_labels[index]

class TestPillDataset(data.Dataset):
    def __init__(self, X: List, img_size: int, transform):
        self.image_paths = X
        self.img_size = img_size
        self.__image_transformer = transform

    def read_image(self, index: int):
        image = cv2.imread(self.image_paths[index], cv2.COLOR_BGR2RGB)
        image = resize_image(image, self.img_size)
        return self.__image_transformer(Image.fromarray(image.astype(np.uint8)))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img = self.read_image(index)
        return img