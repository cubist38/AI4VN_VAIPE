import json
from typing import Dict, Tuple
import pandas as pd

def csv_to_coco(df: pd.DataFrame) -> Tuple[str, Dict]:
    '''
        Generate COCO format file for our labels (annotations)
        
        Args:
            - `df`: A dataframe summarizes the labels
            Columns: `image_id, width, height, x, y, w, h, label
        
        Returns: A relative path to created json file (this path starts from the same root folder of `path`)
    '''
    images = []
    categories = []
    annotations = []

    cnt = 0
    image_id_map = {}
    for name in df['image_id']:
        if name in image_id_map:
            continue
        image_id_map[name] = cnt
        cnt += 1

    obj_id_cnt = 0
    
    def image(row):
        image = {}
        image["file_name"] = row.image_id
        image["id"] = image_id_map[row.image_id]
        image["height"] = row.height
        image["width"] = row.width
        image["depth"] = 3
        return image

    def annotation(row):
        global obj_id_cnt

        annotation = {}
        annotation["image_id"] = image_id_map[row.image_id]

        annotation["bbox"] = [row.x, row.y, row.w, row.h]
        annotation["area"] = row.w * row.h

        annotation["category_id"] = row.class_id
        
        annotation["segmentation"] = None
        annotation["segmented"] = None
        annotation["pose"] = None
        annotation["difficult"] = None
        annotation["iscrowd"] = 0

        return annotation

    for row in df.itertuples():
        anno = annotation(row)
        anno['id'] = obj_id_cnt
        obj_id_cnt += 1
        annotations.append(anno)

    imagedf = df.drop_duplicates(subset=['image_id']).sort_values(by='image_id')
    for row in imagedf.itertuples():
        images.append(image(row))

    class_names = ['pill_class_0', 'pill_class_1', 'pill_class_2', 'pill_class_3', 'pill_class_4', 'pill_class_5', 'pill_class_6', 'pill_class_7', 'pill_class_8', 'pill_class_9', 'pill_class_10', 'pill_class_11', 'pill_class_12', 'pill_class_13', 'pill_class_14', 'pill_class_15', 'pill_class_16', 'pill_class_17', 'pill_class_18', 'pill_class_19', 'pill_class_20', 'pill_class_21', 'pill_class_22', 'pill_class_23', 'pill_class_24', 'pill_class_25', 'pill_class_26', 'pill_class_27', 'pill_class_28', 'pill_class_29', 'pill_class_30', 'pill_class_31', 'pill_class_32', 'pill_class_33', 'pill_class_34', 'pill_class_35', 'pill_class_36', 'pill_class_37', 'pill_class_38', 'pill_class_39', 'pill_class_40', 'pill_class_41', 'pill_class_42', 'pill_class_43', 'pill_class_44', 'pill_class_45', 'pill_class_46', 'pill_class_47', 'pill_class_48', 'pill_class_49', 'pill_class_50', 'pill_class_51', 'pill_class_52', 'pill_class_53', 'pill_class_54', 'pill_class_55', 'pill_class_56', 'pill_class_57', 'pill_class_58', 'pill_class_59', 'pill_class_60', 'pill_class_61', 'pill_class_62', 'pill_class_63', 'pill_class_64', 'pill_class_65', 'pill_class_66', 'pill_class_67', 'pill_class_68', 'pill_class_69', 'pill_class_70', 'pill_class_71', 'pill_class_72', 'pill_class_73', 'pill_class_74', 'pill_class_75', 'pill_class_76', 'pill_class_77', 'pill_class_78', 'pill_class_79', 'pill_class_80', 'pill_class_81', 'pill_class_82', 'pill_class_83', 'pill_class_84', 'pill_class_85', 'pill_class_86', 'pill_class_87', 'pill_class_88', 'pill_class_89', 'pill_class_90', 'pill_class_91', 'pill_class_92', 'pill_class_93', 'pill_class_94', 'pill_class_95', 'pill_class_96', 'pill_class_97', 'pill_class_98', 'pill_class_99', 'pill_class_100', 'pill_class_101', 'pill_class_102', 'pill_class_103', 'pill_class_104', 'pill_class_105', 'pill_class_106', "Out-of-discription"]
    for name in class_names:
        category = {}
        category["supercategory"] = None
        category["id"] = class_names.index(name)
        category["name"] = name
        categories.append(category)

    data_coco = {}
    data_coco["images"] = images
    data_coco["categories"] = categories
    data_coco["annotations"] = annotations

    path = './train_coco.json'
    json.dump(data_coco, open(path, "w"))

    return path, image_id_map



def results_to_coco(results: pd.DataFrame, image_id_map: Dict):
    '''
        Generate COCO format file for our final results

        Args:
            - `results`: A dataframe summarizes the results
            Columns: `image_name, class_id, confidence_score, x_min, y_min, x_max, y_max`
            - `image_id_map`: A dictionary for mapping image name and id
        
        Return:
            A relative path to created json file
    '''
    coco_results = []
    lines = results.values
    for line in lines:
        image_name, class_id, confidence_score, x_min, y_min, x_max, y_max = line
        image_id = image_id_map[image_name]
        width = x_max - x_min
        height = y_max - y_min
        tmp = {
            'image_id': image_id,
            'category_id': class_id,
            'bbox': [x_min, y_min, width, height],
            'score': confidence_score
        }
        coco_results.append(tmp)
    
    path = './coco_results.json'
    with open(path, 'w') as f:
        json.dump(coco_results, f)

    return path