from .PaddleOCR.tools.infer.predict_det import TextDetector
from .PaddleOCR.tools.infer import utility
from .PaddleOCR.ppocr.utils.utility import *
from functools import cmp_to_key

def box_cmp(box1, box2):
    if box1 == box2:
        return 0

    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    if (ymax1 < ymax2) or (ymax1 == ymax2 and xmax1 < xmax2):
        return 1

    return -1

class MyDetector:
    def __init__(self, args: dict):
        '''
        Arguments:
        `args` is a dictionary about the arguments of model, including:
            - `det_model_dir`: path to pretrained model
            ...
        '''
        self.args = utility.parse_args()
        self.args.det_model_dir = args['det_model_dir']
        self.text_detector = TextDetector(self.args)

    def __call__(self, image_dir: str) -> list:
        """
        Arguments:
            - `image_dir`: path to a specific file or folder of images
            - `det_model_dir`: path to pretrained model

        Return: a list of tuple (image path, [boxes]), each box is represented with (xmin, ymin, xmax, ymax)
        """
        box_results = []

        image_file_list = get_image_file_list(image_dir)
        for image_file in image_file_list:
            img, flag = check_and_read_gif(image_file)
            if not flag:
                img = cv2.imread(image_file)

            if img is None:
                continue

            dt_boxes, _ = self.text_detector(img)
            boxes = []
            for box in dt_boxes:
                xmin, ymin = int(box[0][0]), int(box[0][1])
                xmax, ymax = int(box[1][0]), int(box[2][1])
                # Skip too small box
                if xmax - xmin <= 10:
                    continue
                boxes.append((xmin, ymin, xmax, ymax))
            # Sort boxes (it help us validate the drugname)
            boxes.sort(key=cmp_to_key(box_cmp))
            
            box_results.append((image_file, boxes))
        
        return box_results