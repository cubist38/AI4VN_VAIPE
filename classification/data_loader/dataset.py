import torch.utils.data as data
from typing import List
import cv2
from PIL import Image
import numpy as np
from ..utils import resize_image

class PillDataset(data.Dataset):
    def __init__(self, X: List, y: List, img_size: int, transform):
        self.image_paths = X
        self.image_labels = y
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
        return img, self.image_labels[index]