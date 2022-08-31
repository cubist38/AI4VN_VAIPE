import cv2
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

    image = cv2.resize(image, (desired_size, desired_size))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    # padding image with black pixel
    resized = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value = [0, 0, 0])

    return image

def get_lr(optimizer):
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])
    return lr_list