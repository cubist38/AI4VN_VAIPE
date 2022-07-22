import cv2
import numpy as np

DRUG_DELIM = '.-)]} '

def pres_ocr_visualize(img_path: str, boxes: list, drugnames: str) -> np.array:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for idx, box in enumerate(boxes):
        color = (0, 0, 255)

        # box
        xmin, ymin, xmax, ymax = box
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

        # text
        img = cv2.putText(img, drugnames[idx], (xmin, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)

    return img

def rescale_image(img_data: np.array, ratio_w: float, ratio_h: float) -> np.array:
    h, w, _ = img_data.shape
    new_h = int(h * (1 + ratio_h))
    new_w = int(w * (1 + ratio_w))
    return cv2.resize(img_data, (new_w, new_h), cv2.INTER_AREA)

def crop_image(img_data: np.array, xmin: int, ymin: int, xmax: int, ymax: int, delta_w: int = 0, delta_h: int = 0) -> np.array:
    return img_data[ymin - delta_h:ymax + delta_h, xmin - delta_w:xmax + delta_w].copy()

def is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except:
        return False

def is_quantity(s: str) -> bool:
    key_words = ['sl', 'số lượng', 'viên', 's.Lượng']
    ss = s.lower()

    for word in key_words:
        if ss.find(word) != -1:
            return True

    return False

def is_drugname(before: str, text: str, after: str) -> bool:
    criteria_count = 0

    # Start with <int>, followed by a character in DRUG_DELIM
    i = 0
    while i < len(text) and text[i].isdigit():
        i += 1
    if i < len(text) and text[i] in DRUG_DELIM:
        criteria_count += 1

    # Contain 'mg'
    if text.lower().find('mg') != -1:
        criteria_count += 1
    
    # before or after contains 'SL', 'Số lượng', 'Viên'
    if is_quantity(before) or is_quantity(after):
        criteria_count += 1

    return True if criteria_count >= 2 else False

def get_drugname(text: str) -> str:
    i = 0
    while i < len(text) and (text[i].isdigit() or text[i] in DRUG_DELIM):
        i += 1
    return text[i:] if i < len(text) else ''