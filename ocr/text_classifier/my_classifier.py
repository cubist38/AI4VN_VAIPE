import sys
import os
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(__dir__, 'vietocr'))

from .vietocr.vietocr.tool.predictor import Predictor
from .vietocr.vietocr.tool.config import Cfg
from PIL import Image
import numpy as np
from utils.prescription import *

class MyClassifier():
    def __init__(self, args: dict):
        '''
        Arguments:
        `args` is a dictionary about the arguments of model, including:
            - 'cls_model_name`: name of pretrained model
            - `cls_model_dir`: path to pretrained model
        '''
        self.config = Cfg.load_config_from_name(args['cls_model_name'])
        self.config['weights'] = args['cls_model_dir']
        self.config['cnn']['pretrained'] = False
        self.config['device'] = 'cuda:0'
        self.config['predictor']['beamsearch'] = False
        self.classifier = Predictor(self.config)

    def __call__(self, cv2_image: np.array) -> str:
        '''
        Arguments:
            -`cv2_image`: image read by opencv

        Return: text
        '''
        new_img = rescale_image(cv2_image, ratio_w=0.05, ratio_h=0.04)
        texts = self.classifier.predict(Image.fromarray(new_img))
        return texts