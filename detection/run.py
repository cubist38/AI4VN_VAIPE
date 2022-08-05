import torch
import os
from typing import Dict
from PIL import Image

def do_detection(image_folder: str, batch_size: int = 32, model_name: str = 'yolov5') -> Dict:
    '''
        Run Object Detection model to return the result of images in `image_folder`

        Returns:
            A dictonary with:
            - key: `path_to_image`
            - value: a list of objects with the following information: `(xmin, ymin, xmax, ymax, label)`
                where `label` is a number between 0 and 121.
    '''
    image_files = os.listdir(image_folder)
    image_batches = []
    for i in range(len(image_files) // batch_size + 1):
        start = i * batch_size
        end = min(len(image_files), (i + 1) * batch_size)
        if start > end:
            break
        image_batches.append(image_files[start:end])

    # model = torch.hub.load('./detection/yolo/yolov5', 'custom', path='./detection/yolo/yolov5/runs/train/exp/yolov5_best.pt', source='local')
    results = {}

    if model_name == 'yolov5':
        model = torch.hub.load('./detection/yolo/yolov5', 'custom', path='./detection/yolo/yolov5/runs/train/exp/yolov5_best.pt', source='local')
        
        # Batch inference
        for batch in image_batches:
            image_data = []
            for file in batch:
                path = os.path.join(image_folder, file)
                image_data.append(Image.open(path))

            outputs = model(image_data)
            dfs = outputs.pandas().xyxy
            for idx, df in enumerate(dfs):
                path = os.path.join(image_folder, batch[idx])
                xmins, ymins, xmaxs, ymaxs = df['xmin'], df['ymin'], df['xmax'], df['ymax']
                confs = df['confidence']
                labels = df['class']
                boxes = []
                for i in range(len(xmins)):
                    xmin, ymin, xmax, ymax = int(xmins[i]), int(ymins[i]), int(xmaxs[i]), int(ymaxs[i])
                    conf, label = confs[i], labels[i]
                    boxes.append((xmin, ymin, xmax, ymax, label, conf))
                results[path] = boxes

        # Single inference
        # image_files = os.listdir(image_folder)
        # for file in image_files:
        #     path = os.path.join(image_folder, file)
        #     outputs = model(path)
        #     df = outputs.pandas().xyxy[0]
        #     xmins, ymins, xmaxs, ymaxs = df['xmin'], df['ymin'], df['xmax'], df['ymax']
        #     confs = df['confidence']
        #     labels = df['class']
        #     boxes = []
        #     for i in range(len(xmins)):
        #         xmin, ymin, xmax, ymax = int(xmins[i]), int(ymins[i]), int(xmaxs[i]), int(ymaxs[i])
        #         conf, label = confs[i], labels[i]
        #         boxes.append((xmin, ymin, xmax, ymax, label, conf))
        #     results[path] = boxes
        
    else:
        print('This model_name is not valid')
        return None
    
    return results