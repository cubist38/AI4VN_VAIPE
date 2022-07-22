# AI4VN_VAIPE

## Visualizer
### 1. Pills
```python
from utils.io import read_image, read_bbox
from visualizer.pill import apply_bbox
import cv2

image_path = 'data/public_train/pill/image/VAIPE_P_0_0.jpg'
bbox_path = 'data/public_train/pill/label/VAIPE_P_0_0.json'

image = read_image(image_path)
bbox_list = read_bbox(bbox_path)

bbox_image = apply_bbox(image, bbox_list)

cv2.imshow('visualize pills', bbox_image)
cv2.waitKey(0)

#closing all open windows 
cv2.destroyAllWindows() 
```

## Drugname OCR
### 1. Download pre-trained weights

- Firstly, you have to download <a href="https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar">text detector weights</a> from PaddleOCR. Then extract it to this path: `./ocr/text_detector/PaddleOCR/weights`.
- After that, download <a href="https://drive.google.com/uc?id=1ePh6kXJtnAUG7zqXixUEEor54uVnEY4k">text classifier weights</a> (fine-tuning from pre-trained model of vietocr) and put it in this path: `./ocr/text_classifier/vietocr/weights`.

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