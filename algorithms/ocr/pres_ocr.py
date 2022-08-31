import cv2
import os

from algorithms.ocr.text_detector.my_detector import MyDetector
from algorithms.ocr.text_classifier.my_classifier import MyClassifier
from utilities.prescription import *

def pres_ocr(image_dir: str, saved: bool = False) -> list:
    '''
    Arguments:
        - `image_dir`: path to a specific image file or folder

    Return: a list of tuple (image path, list of drugname)
    '''
    if saved:
        result_folder = 'ocr/ocr_result'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

    args = {
        'det_model_dir': './weights/ocr/ch_PP-OCRv3_det_infer',
        'cls_model_dir': './weights/ocr/seq2seq_finetuned.pth',
        'cls_model_name': 'vgg_seq2seq'
    }
    
    detector = MyDetector(args)
    detect_result = detector(image_dir)

    classifier = MyClassifier(args)
    ocr_result = []
    for (image_path, boxes) in detect_result:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None:
            continue

        text_of_image = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            cropped = crop_image(image, xmin, ymin, xmax, ymax, delta_w=6, delta_h=3)
            try:
                text = classifier(cropped).rstrip()
                text_of_image.append(text)
            except:
                continue

        drugnames = []
        filtered_boxes = []
        n = len(text_of_image)
        for i in range(n):
            before = '' if i == 0 else text_of_image[i - 1]
            text = text_of_image[i]
            after = '' if i == n - 1 else text_of_image[i + 1]
            if is_drugname(before, text, after):
                name = get_drugname(text)
                if len(name) == 0:
                    continue
                drugnames.append(name)
                if saved:
                    filtered_boxes.append(boxes[i])

        tmp = image_path.replace('\\', '/')
        image_name = tmp.split('/')[-1]
        ocr_result.append((image_name, drugnames))

        if saved:
            out_img = pres_ocr_visualize(image_path, filtered_boxes, drugnames)
            img_name_pure = os.path.split(image_path)[-1]
            out_path = os.path.join(result_folder,
                                    "OCR_{}".format(img_name_pure))
            cv2.imwrite(out_path, out_img)

    return ocr_result

# if __name__ == '__main__':
#     ocr_result = pres_ocr('./images', saved=False)
#     for item in ocr_result:
#         image_path, drugnames = item
#         print(image_path)
#         for drug in drugnames:
#             print(drug)
#         print('\n-----------------------------------------------------')
    
