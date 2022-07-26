import cv2
import os
import shutil
import numpy as np

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

    # padding image with black pixel
    resized = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value = [0, 0, 0])

    return resized


def gen_cluster_result(dst_folder: str, image_paths: list, labels: list) -> None:
    '''
        Generate image folders based on clustering result

        Args:
            - `dst_folder`: Path to the directory containing the results
            - `clustering_output`: a list of N list, each list contains the paths of images in the corresponding cluster
    '''
    if os.path.exists(dst_folder):
        os.remove(dst_folder)
    
    os.mkdir(dst_folder)

    for i in range(len(image_paths)):
        path = image_paths[i]
        label = labels[i]
        folder = os.path.join(dst_folder, str(label))
        if not os.path.exists(folder):
            os.mkdir(folder)
        name = path.split('/')[-1]
        shutil.copy(path, os.path.join(folder, name))