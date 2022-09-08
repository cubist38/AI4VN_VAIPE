# AI4VN_VAIPE - Team Real Cotton Candy

Content:
- <a href="#docker">1. Docker command</a>
- <a href="#training">2. Training</a>
    - <a href="#ocr_train">2.1. OCR</a>
    - <a href="#yolov5_train">2.2. YOLOv5</a>
    - <a href="#cls_train">2.3. Classification</a>
- <a href="#inference">3. Inference</a>
    - <a href="#trained weights">3.1. Download trained weights</a>
    - <a href="#result">3.2. Generate results</a>

## 1. Docker command
<span id="docker"></span>

- Build docker image:
```
docker build -t ai4vn-teamrealcottoncandy .
```

- Run docker container:
```
docker run ai4vn-teamrealcottoncandy
```

- Mount folder in docker container to real folder (change the source folder to a appropriate folder):
```
docker run -d -it --name ai4vn-teamrealcottoncandy --mount source=/mnt/disk1/ai4vn-teamrealcottoncandy, target=/workspace
```

## 2. Training
<span id="training"></span>

### 2.1. OCR
<span id="ocr_train"></span>

- Download [OCR training data](https://drive.google.com/uc?id=1TQF7RIvhM6VljmDgFnGmnfALls-NlsOY&export=download), then extract it to the folder `training/ocr/vietocr_data`. Notice that, the images in this dataset is cropped from the original prescription images of VAIPE contest. After this step, this folder should be in this format:
    ```
    ---training/ocr/vietocr_data/
        |---images/
        |---train_annotations.txt
        |---valid_annotations.txt
    ```

- Run the OCR training with this following command:
    ```
    python training/ocr/train_ocr.py
    ```

- When the training finished successfully, the weights will be stored at `seq2seq_finetuned.pth`. To this it for inference, let's copy it to this folder: `weights/ocr/seq2seq_finetuned.pth`.

### 2.2. YOLOv5
<span id="yolov5_train"></span>

- We will training 2 version of YOLOv5, with different types of dataset:
    - Download [training data with 107 class](https://drive.google.com/uc?id=1frC6SLsAh6eQ5TW6R_OpXTTBc72RoNq_&export=download), then extract it to the folder `training/detection/yolo_107/yolo_107_data`. We created this dataset by cutting out the pill of class id 107 (out of prescription) from the images (so we will have 107 class id, from 0 to 106). After this step, this folder should be in this format:
    ```
    ---training/detection/yolo_107/yolo_107_data/
        |---train/
        |---val/
    ```
    - Follow the similar step with the [training data with 1 class](https://drive.google.com/uc?id=1Qs-4YWKkzlsNtiJvlIgh0lvpXZXcHlyz&export=download) and folder `training/detection/yolo_1/yolo_1_data`. This dataset is created by replacing the class_id of all pills with 0 (use YOLOv5 for bounding box detection).

- Run the following commands to train YOLOv5 models:

    - YOLOv5 with 107 class:
    ```
    python training/detection/yolo_107/train.py --img 640 --batch 16 --epochs 100 --data training/detection/yolo_107/vaipe.yaml --weights yolov5s.pt
    ```

    - YOLOv5 with 1 class:
    ```
    python training/detection/yolo_1/train.py --img 640 --batch 16 --epochs 100 --data training/detection/yolo_1/vaipe.yaml --weights yolov5s.pt
    ```

- When the trainings finished successfully, the weights will be stored at these folders:
    - `training/detection/yolo_107/runs/exp/train/weights/best.pt` for YOLOv5 with 107 class. We move it to the path `weights/detection/yolov5_weights_with_label.pt` for inference (that weights file was renamed).
    - `training/detection/yolo_1/runs/exp/train/weights/best.pt` for YOLOv5 with 1 class. Do the similar steps to the previous with the file name `yolov5_weights_without_label.pt`.

### 2.3. Classification
<span id="cls_train"></span>

- Download the [training data](https://drive.google.com/uc?id=1Lvr8AOnqMfAP9bdWTLbiirtIFEQPxu1M&export=download) and extract it to the folder `training/classification/cls_data`. We created this dataset by cropping the pills from original images of VAIPE contest. After this step, this folder should be in this format:

    ```
    ---training/classification/cls_data/
        |---single_pill/
        |---single_pill.json
    ```

- Run the following command:
    ```
    python training/classification/train_cls.py
    ```
## 3. Inference
<span id="inference"></span>

For inference, we can use the weights from section <a href="#training">2. Training</a>, or download the weights from our Drive.

***Note:*** We have also included these weights in the docker container. If you can't find the `weights` folder, let's follow section <a href="#trained weights">3.1. Download trained weights</a>.

### 3.1. Download trained weights
<span id="trained weights"></span>

- **OCR weights**:
    - Download <a href="https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar">text detector weights</a> from PaddleOCR and **extract** it, then download <a href="https://drive.google.com/uc?id=1O5DkqiM3lE50sjzVz5_NuguILaS4BUER">text classifier weights</a> (fine-tuning from pre-trained model of vietocr).
    - Put these weights in the path `weights/ocr/`. After these steps, folder `weights/ocr` should be in this format:
    ```
    ---weights/ocr/
        |---ch_PP-OCRv3_det_infer/
            |--- files
        |---seq2seq_finetuned.pth
    ```  

- **Pill detection**:
    - First, you have to download [YOLOv5 weights without label](https://drive.google.com/file/d/1JzCyoExM7PB-wU9eNLENokkABDJRe76F/view?usp=sharing) and [YOLOv5 weights with label](https://drive.google.com/file/d/1BaQ_fBSYFyB0u9bm3HEq-3mC4G7cY_fG/view?usp=sharing). After that, you should put them in this path: `weights/detection/`.

- **Pill classification**:
    <!-- - `Swin Transformer V2` can be found at [Google drive](https://drive.google.com/drive/folders/1x7TsyX7xj_wRFAwEzgJ8omGGS9MuWNnZ?usp=sharing).  
    Download and move to `weights/cls/`. After that, we have the path `weights/cls/swinv2_kfold`.
    - `Swin Tiny` can be found at [Google drive](https://drive.google.com/drive/folders/1ZPixqk1kqinfLFxT45RA2A3rDekjUxN0?usp=sharing).  
    Download and move to `weights/cls/`. After that, we have the path `weights/cls/swin_tiny_kfold`. -->
    - Download the entire weight folders of [Swin Transformer V2](https://drive.google.com/drive/folders/16M99KvYmC66fQty3PvtXDAdmm9W61DSO?usp=sharing) and [Swin Tiny](https://drive.google.com/drive/folders/1eWLflWQ5LISuU-d7XEqbwS-oYfhwJB5a?usp=sharing), then put them in this path `weights/cls/`.
    - After that, the folder `weights/cls/` should be in this format:
    ```
    ---weights/cls/
        |---swinv2_kfold/
            |--- *.pth
        |---swin_tiny_kfold/
            |--- *pth
    ```

### 3.2. Generate results
<span id="result"></span>

The data folder `data_path`'s structure has to be in the following format:

```
---data_path/
    |---pill/
        |---image/
    |---prescription/
        |---image/
    |---pill_pres_map.json
```

Before running the generate results script, we have to specify the path to data folder in the file `configs/config_inference.yaml`, at these 3 lines (the following values are for illustration):
```
pill_image_dir: data_path/pill/image
pres_image_dir: data_path/prescription/image
pill_pres_map_path: data_path/pill_pres_map.json
```

Currently, we set `data_path` to `data/public_test`.    

To generate result, we run the following command:
```
python generate_results.py
```

The result file will be stored at `results/csv/results.csv`.
