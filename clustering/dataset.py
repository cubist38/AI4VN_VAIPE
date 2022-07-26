import torch.utils.data as data
from typing import List
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np

from .utils import resize_image

class PillDataset(data.Dataset):
    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths
        self.__image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def read_image(self, index: int):
        image = cv2.imread(self.image_paths[index], cv2.COLOR_BGR2RGB)
        image = resize_image(image, 224)
        return self.__image_transformer(Image.fromarray(image.astype(np.uint8)))

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, index: int):
        img = self.read_image(index)
        return self.image_paths[index], img 