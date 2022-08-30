from typing import Dict

# import some common libraries
import cv2
import matplotlib.pyplot as plt
import os

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def get_mask(yaml_cfg: Dict, image_file: str):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(yaml_cfg['model']))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = yaml_cfg['num_classes']
    cfg.MODEL.WEIGHTS = os.path.join(yaml_cfg['model_path'])
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = yaml_cfg['threshold']
    predictor = DefaultPredictor(cfg)
    img = cv2.imread(image_file)
    masks = predictor(img)["instances"].to("cpu").pred_masks.numpy()
    return masks