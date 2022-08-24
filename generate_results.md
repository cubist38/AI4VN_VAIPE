# AI4VN_VAIPE

Content:
- <a href="#trained weights">Download trained weights</a>
- <a href="#result">Generate results</a>

## Download trained weights
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
    - Download [YOLOv5 weights](https://drive.google.com/uc?id=1Eiwp6vd5wK1Fu_Wxuowpo5nx2DUhcyzq) and put it in the path: `weights/detection/`.

- **Pill classification**:
    <!-- - `Swin Transformer V2` can be found at [Google drive](https://drive.google.com/drive/folders/1x7TsyX7xj_wRFAwEzgJ8omGGS9MuWNnZ?usp=sharing).  
    Download and move to `weights/cls/`. After that, we have the path `weights/cls/swinv2_kfold`.
    - `Swin Tiny` can be found at [Google drive](https://drive.google.com/drive/folders/1ZPixqk1kqinfLFxT45RA2A3rDekjUxN0?usp=sharing).  
    Download and move to `weights/cls/`. After that, we have the path `weights/cls/swin_tiny_kfold`. -->
    - Download the entire weight folders of [Swin Transformer V2](https://drive.google.com/drive/folders/1x7TsyX7xj_wRFAwEzgJ8omGGS9MuWNnZ?usp=sharing) and [Swin Tiny](https://drive.google.com/drive/folders/1ZPixqk1kqinfLFxT45RA2A3rDekjUxN0?usp=sharing), then put them in the path `weights/cls/`.
    - After that, the folder `weights/cls/` should be in this format:
    ```
    ---weights/cls/
        |---swinv2_kfold/
            |--- *.pth
        |---swin_tiny_kfold/
            |--- *pth
    ```

## Generate results
<span id="result"></span>


To generate the 'results.csv' file for data in folder `data_path` whose structure has to be in the following format:


```
---data_path/
    |---pill/
        |---image/
    |---prescription/
        |---image/
    |---pill_pres_map.json
```

To run this code, we must do:
- Firstly, run `requirements.txt` file.
    ```
    pip install -r requirements.txt
    ```
- Then, run `generate_results.py` file.
    ```
    python generate_results.py
    ```
