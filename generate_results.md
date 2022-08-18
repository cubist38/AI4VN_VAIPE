# AI4VN_VAIPE

Content:
- <a href="#trained weights">Dataset</a>
- <a href="#result">Generate results</a>

## Download trained weights
<span id="trained weights"></span>
- `Swin Transformer V2` can be found at [Google drive](https://drive.google.com/drive/folders/1x7TsyX7xj_wRFAwEzgJ8omGGS9MuWNnZ?usp=sharing).  
    Download and move to `weights/cls/`. After that, we have the path `weights/cls/swinv2_kfold`.

- `Swin Tiny` can be found at [Google drive](https://drive.google.com/drive/folders/1ZPixqk1kqinfLFxT45RA2A3rDekjUxN0?usp=sharing).  
    Download and move to `weights/cls/`. After that, we have the path `weights/cls/swin_tiny_kfold`.


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
- Finally, run `generate_results.py` file.
    ```
    python generate_results.py
    ```
