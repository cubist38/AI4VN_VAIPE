from typing import Dict
import os
import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from model import MobilenetV3
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from config import *


# Transform the image, so it becomes readable with the model
transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def extract_features(image_path: str, feature_extractor: nn.Module):
    image = cv2.imread(image_path)
    image = transformer(image)
    image = image.reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    image = image.to(device)
    with torch.no_grad():
        feature = feature_extractor(image)
    # convert to nparray and reshape it
    feature = feature.cpu().detach().numpy().reshape(-1)
    return feature


def do_clustering(image_src: str, image_list: list, feature_extractor: nn.Module) -> list:
    '''
        Args:
            image_src: path to image folder
            image_list: name of images
        Return:
            clustering_output = N x [] 
            where
                - N is the number of clusters
                - each group contains a list of image path belong to that group
    '''
    image_paths = np.array([os.path.join(image_src, image_name)
                           for image_name in image_list])

    # Feature extraction
    data = {}
    for path in image_paths:
        data[path] = extract_features(path, feature_extractor)

    # PCA to 150-dimension
    features = [feature for feature in data.values()]
    pca = PCA(n_components=150)
    pca.fit(features)
    features = pca.transform(features)

    return None

    # # DBSCAN
    # dbscan = DBSCAN(eps=THRES, min_samples=MINPTS + 1, metric='euclidean')
    # clusters = dbscan.fit_predict(features)

    # clustering_output = []
    # # noise image will have cluster = -1, we consider it as a independent class
    # noise_paths = image_paths[clusters == -1]
    # for path in noise_paths:
    #     clustering_output.append([path])
    # # for "real" cluster:
    # for cluster_id in range(0, max(clusters)):
    #     cluster_paths = image_paths[clusters == cluster_id]
    #     clustering_output.append(list(cluster_paths))

    # return clustering_output

# if __name__ == '__main__':
#     feature_extractor = MobilenetV3(option='large', pretrained=True)
#     feature_extractor.to(device)

#     base_path = 'data/personal_new_pill/image'
#     image_name = 'VAIPE_P_0_0_0.jpg'
#     path = os.path.join(base_path, image_name)
#     features = extract_features(path, feature_extractor)
#     print(features.shape)
