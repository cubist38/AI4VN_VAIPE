from evaluate.wmap import compute_wmap
from evaluate.coco_format import *
import pandas as pd
import os

def eval(results_path: str, train_path: str):
    '''
        Evaluate the results with wmAP metrics

        Args:
            - `results_path`: Path to results file (.csv), with the following columns: `image_name, class_id, confidence_score, x_min, y_min, x_max, y_max`
            - `train_path`: Path to train.csv file (currently in data/tran.csv)

    '''
    results = pd.read_csv(results_path)
    train_df = pd.read_csv(train_path)

    images = results['image_name']
    train_df = train_df[train_df['image_id'].isin(images)]

    anno_path = csv_to_coco(train_df)
    pred_path = results_to_coco(results)
    
    wmap50, wmap = compute_wmap(anno_path, pred_path)
    print('wmAP50:', wmap50)
    print('wmAP:', wmap)

    os.remove(anno_path)
    os.remove(pred_path)
