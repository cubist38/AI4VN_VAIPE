from algorithms.ocr.pres_ocr import pres_ocr

from typing import Dict
import json

def run_ocr(image_dir: str, output_dir: str) -> Dict:
    ocr_result = pres_ocr(image_dir=image_dir)
    ocr = {}
    
    for l in ocr_result:
        ocr[l[0]] = l[1]
        
    with open(output_dir, 'w', encoding='utf8') as f:
        json.dump(ocr, f, ensure_ascii=False)
