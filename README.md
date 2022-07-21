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

- Firstly, you have to download <a href="https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar">text detector weights</a> from PaddleOCR. Then extract it to this path: `./ocr/text_detector/PaddleOCR/weights/ch_PP-OCRv3_det_infer`.
- After that, download <a href="https://drive.google.com/uc?id=1nTKlEog9YFK74kPyX0qLwCWi60_YHHk4">text classifier weights</a> from vietocr and put it in this path: `'./ocr/text_classifier/vietocr/weights/transformerocr.pth'`.

### 2. Run OCR for prescription

```python
from ocr.pres_ocr import pres_ocr

# image_dir can be a path to specify image or a folder of images
ocr_result = pres_ocr(image_dir='./images', saved=False)
for item in ocr_result:
    image_path, drugnames = item
    print(image_path)
    for drug in drugnames:
        print(drug)
    print('-----------------------------------------------------')
```