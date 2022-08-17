from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from .augmentation import build_ins_seg_train_aug
from .utils import get_lettuce_dicts

def register_dataset(train_annotations, test_annotations):
    for d in ["train", "val"]:
        if d == "train":
            DatasetCatalog.register("vaipe_" + d, lambda d=d: get_lettuce_dicts(train_annotations))
            MetadataCatalog.get("vaipe_" + d).set(thing_classes=["pill"])
        else:
            DatasetCatalog.register("vaipe_" + d, lambda d=d: get_lettuce_dicts(test_annotations))
            MetadataCatalog.get("vaipe_" + d).set(thing_classes=["pill"])

class Trainer(DefaultTrainer):

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=build_ins_seg_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

def train(yaml_cfg):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(yaml_cfg['model']))
    cfg.DATASETS.TRAIN = ("vaipe_train", )
    cfg.DATASETS.TEST = ()
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml_cfg['model'])  # Let training initialize from model zoo
    cfg.SOLVER.BASE_LR = yaml_cfg['base_lr']  
    cfg.SOLVER.MAX_ITER = yaml_cfg['max_iter']
    cfg.SOLVER.STEPS = yaml_cfg['steps']

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (lettuce)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    return trainer

def test(trainer, yaml_cfg):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(yaml_cfg['model']))
    cfg.MODEL.WEIGHTS = os.path.join(yaml_cfg['model_path'])
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = yaml_cfg['threshold']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = yaml_cfg['num_classes']
    cfg.DATASETS.TEST = ("vaipe_val", )
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("vaipe_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "vaipe_val")
    inference_on_dataset(trainer.model, val_loader, evaluator)    