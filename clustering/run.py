from clustering.models.swin_transformer import swin_tiny_transformer
from clustering.extractor import get_features
from sklearn.cluster import DBSCAN, KMeans
import os
import torch
import json

def do_clustering(cfg: dict) -> list:
    '''
        Run the clustering algorithm (KMeans at the current time) with the configuration `cfg`

        Returns:
            - `(image_paths, kmean_output)`: Path to each image and its cluster, use for generate image folders 
            based on the clustering result
            -  `detection_data_dict`: Output used for object detection algorithms
            - `label_freq_dict`: Frequency of each original class (from 0 to 107) on each output cluster

    '''
    with open(cfg['data_dict']) as f:
        data_dict = json.load(f)

    device = torch.device(cfg['device'])

    net = swin_tiny_transformer().to(device)
    net.load_state_dict(torch.load(cfg['weight']), strict=False)
    net.eval()

    image_paths = [os.path.join(cfg['img_src'], img_name) for img_name in data_dict.keys()]
    image_labels = list(data_dict.values())

    features = get_features(cfg, net, image_paths)
    
    print('K-means clustering ...', end = ' ')
    kmeans = KMeans(n_clusters=cfg['K'])
    kmean_output = kmeans.fit_predict(features)
    print('Done!')

    new_label_dict = {}
    for id, img_name in enumerate(data_dict.keys()):
        new_label_dict[img_name] = kmean_output[id]
    
    with open(cfg['bbox_dict']) as f:
        bbox_dict = json.load(f)

    annotation_dict = {}
    label_freq_dict = {}
    for item in bbox_dict:
        img_id = item['img_id']
        new_label = int(new_label_dict[img_id])

        if new_label not in label_freq_dict:
            label_freq_dict[new_label] = [0 for _ in range(108)]
        
        label_freq_dict[new_label][item['label']] += 1

        root_img = '_'.join(img_id.split('_')[:-1]) + '.jpg'
        if root_img not in annotation_dict:
            annotation_dict[root_img] = []

        annotation = {}
        annotation['x'] = item['x']
        annotation['y'] = item['y']
        annotation['w'] = item['w']
        annotation['h'] = item['h']
        annotation['label'] = new_label
        annotation_dict[root_img].append(annotation)

    detection_data_dict = []
    for root_img in annotation_dict.keys():
        detection_item = {}
        detection_item['image_id'] = root_img
        detection_item['annotation'] = annotation_dict[root_img]
        detection_data_dict.append(detection_item)

    return (image_paths, kmean_output), detection_data_dict, label_freq_dict