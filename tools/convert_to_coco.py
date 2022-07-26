import datetime
from typing import Dict
from PIL import Image
import json
import argparse
import os

parser = argparse.ArgumentParser(description='Coco convertor')
parser.add_argument('--data_json', default=None, type=str, help='Path to json data')
parser.add_argument('--src_image', default=None, type=str, help='Path to image data')
parser.add_argument('--save', default='data/coco.json', type=str, help='Path to save coco data')
parser.add_argument('--test', default=False, type=bool, help='set True to test coco covertor')
args = parser.parse_args()

def convert(dataset_dicts: Dict, src_image: str, num_classes: int = 122):
    # current an id can have more than 1 name.
    categories = [
        {"id": id, "name": id}
        for id in range(num_classes)
    ]

    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        image_name = image_dict['image_id']
        print(f'Converting on {image_name} ...', end='')
        img = Image.open(os.path.join(src_image, image_name))
        w, h = img.size
        del img # avoid memory leak

        coco_image = {
            "id": image_id,
            "width": w,
            "height": h,
            "file_name": image_dict['image_id'],
        }
        coco_images.append(coco_image)

        for obj in image_dict['annotation']:
            # bbox must be xywh format
            bbox = [obj['x'], obj['y'], obj['w'], obj['h']]

            coco_annotation = {}
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = obj['w'] * obj['h']
            coco_annotation["category_id"] = obj["label"]
            coco_annotation["iscrowd"] = 0

            coco_annotations.append(coco_annotation)
        print(' Done!')

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }

    coco_dict = {
        "info": info,
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
        "licenses": None,
    }

    print('=' * 30)
    print("Conversion finished!")
    print(f"Num images: {len(coco_images)}, num annotations: {len(coco_annotations)}")

    return coco_dict

if __name__ == '__main__':
    data_dicts = None
    with open(args.data_json) as f:
        data_dicts = json.load(f)

    if args.test:
        data_dicts = data_dicts[:10] # only test first 10 images

    coco_dicts = convert(data_dicts, args.src_image)
    with open(args.save, 'w') as f:
        json.dump(coco_dicts, f)
    print(f'Save to {args.save} successfully!')