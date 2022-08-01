# AI4VN_VAIPE

Content:
- <a href="#dataset">Dataset</a>
- <a href="#classification">Pill Classification</a>
- <a href="#ocr">Drugname OCR</a>
- <a href="#detection">Pill detection</a>
- <a href="#evaluation">Evaluation</a>
- <a href="#visualizer">Visualizer</a>


## Dataset
<span id="dataset"></span>
Download the data at [Google drive](https://drive.google.com/drive/folders/1PNhStby1B_xZBwS1mic-EssPX8Q_0odR?usp=sharing) and re-ordering follows the below structure.
```
---data/
    |---label/
        |---single_pill/
        |---single_pill.json
        |---pills/
            |---image/
            |---label/
        |---pills.json
    |--unlabel/             # for inference
```


## Pills classification
<span id="classification"></span>

### 1. Download trained weights
All trained models can be found at [Google drive](https://drive.google.com/drive/folders/1kUopc2ZHbzSY5lTboR7XFtIKtIVdQwAo?usp=sharing).  
Download and move to `weights/cls/`.
### 2. Run classification
(For re-training classification)  
Change the hyper-parameters in coressponding config file `configs/config_cls.yaml`.  
To run training  
```bash
python train_cls.py
```

## Drugname OCR
<span id="ocr"></span>

### 1. Download pre-trained weights

- Firstly, you have to download <a href="https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar">text detector weights</a> from PaddleOCR. Then extract it to this path: `./ocr/text_detector/PaddleOCR/weights`.
- After that, download <a href="https://drive.google.com/uc?id=1O5DkqiM3lE50sjzVz5_NuguILaS4BUER">text classifier weights</a> (fine-tuning from pre-trained model of vietocr) and put it in this path: `./ocr/text_classifier/vietocr/weights`.

### 2. Run OCR for prescription

```python
from ocr.pres_ocr import pres_ocr
import json

# image_dir can be a path to specify image or a folder of images
ocr_result = pres_ocr(image_dir='./personal_test_images', saved=False)

ocr_output_dict = {}
for item in ocr_result:
    image_path, drugnames = item
    ocr_output_dict[image_path] = []
    for drug in drugnames:
        ocr_output_dict[image_path].append(drug)

with open("./personal_pres_ocr_output.json", "w", encoding="utf-8") as f:
    json.dump(ocr_output_dict, f, ensure_ascii=False)
```

## Pill detection
<span id="detection"></span>

### 1. Download weights of detection models

- You have to download [YOLOv5 weights](https://drive.google.com/uc?id=1Eiwp6vd5wK1Fu_Wxuowpo5nx2DUhcyzq) and put it in this path: `./detection/yolo/yolov5/runs/train/exp/`.

### 2. Run object detection for pill images

```python
from detection.run import do_detection

image_folder = './personal_images'

results = do_detection(image_folder, model_name='yolov5')

for image, boxes in results.items():
    print(image)
    print('xmin, ymin, xmax, ymax, label, conf')
    for box in boxes:
        print(box)
    print('----------------------------')
```

## Evaluation
<span id="evaluation"></span>

After you generate `results.csv` file based on the script at `results/result_process.ipynb`, let's run the following code to get the wmAP metrics of your result:

```python
from evaluate.run import eval
eval('path/to/your/results.csv', 'data/train.csv')
```

## Visualizer
<span id="visualizer"></span>

```python
from utils.io import read_image, read_bbox
from visualizer.pill import apply_bbox
import cv2

image_path = 'data/label/pill/image/VAIPE_P_0_0.jpg'
bbox_path = 'data/label/pill/label/VAIPE_P_0_0.json'

image = read_image(image_path)
bbox_list = read_bbox(bbox_path)

bbox_image = apply_bbox(image, bbox_list)

cv2.imshow('visualize pills', bbox_image)
cv2.waitKey(0)

#closing all open windows 
cv2.destroyAllWindows() 
```