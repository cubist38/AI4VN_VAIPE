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