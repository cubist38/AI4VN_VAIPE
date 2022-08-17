import pandas as pd
import cv2
import os
import numpy as np

IMAGE_PATH = 'data/public_test/pill/image'
RESULT_PATH = 'results/csv/results.csv'

def resize_image(image: np.array, desired_size: int) -> np.array:
    '''
        Resize image to keep-ratio square image
        ----------
        Parameters
        ----------
            image: np.array # original image (H x W x 3)
            desired_size: int # size of image to be resized
        ----------
        Return
        ----------
            resized_image: np.array # (desired_size x desired_size x 3)
    '''
    old_size = image.shape[:2] # (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    return image, new_size

if __name__ == '__main__':
    df = pd.read_csv(RESULT_PATH)
    image_list = os.listdir(IMAGE_PATH)

    for img_name in image_list:
        print('Processing', img_name)
        img_path = os.path.join(IMAGE_PATH, img_name)
        df_tmp = df[df['image_name'] == img_name]
        if len(df_tmp) == 0:
            print('--------------------')
            continue
        
        image = cv2.imread(img_path)
        H, W = image.shape[:2]
        image, (h, w) = resize_image(image, 640)

        df_tmp = df_tmp.reset_index()
        for i in range(len(df_tmp)):
            x_min = int(df_tmp['x_min'][i] / W * w)
            y_min = int(df_tmp['y_min'][i] / H * h)
            x_max = int(df_tmp['x_max'][i] / W * w)
            y_max = int(df_tmp['y_max'][i] / H * h)
            print(x_min, y_min, x_max, y_max)

            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        

        cv2.imwrite('data/public_test_detection_result/' + img_name, image)
        