from detectron2.data import transforms as T

def build_ins_seg_train_aug(cfg):
    train_aug = [T.ResizeShortestEdge(
                            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
                        ),
                      T.RandomFlip(prob=0.5),
                      T.RandomFlip(prob=0.5, horizontal = False, vertical=True),
                      T.RandomBrightness(0.8, 1.2),
                      T.RandomLighting(0.8)
                      ]
    return train_aug