from clustering.models.swin_transformer import swin_tiny_transformer
from clustering.extractor import get_features
from sklearn.cluster import DBSCAN, KMeans
import os
import torch
import yaml
import json
import shutil

def gen_cluster_result(dst_folder: str, clustering_output: list):
    if os.path.exists(dst_folder):
        os.remove(dst_folder)
    else:
        os.mkdir(dst_folder)

    for id in range(len(clustering_output)):
        folder = os.path.join(dst_folder, str(id))
        if not os.path.exists(folder):
            os.mkdir(folder)
        for path in clustering_output[id]:
            name = path.split('/')[-1]
            shutil.copy(path, os.path.join(folder, name))

if __name__ == '__main__':
    cfg = yaml.safe_load(open('configs/config_clustering.yaml'))
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

    with open(cfg['detection_data_path'], 'w') as f:
        json.dump(detection_data_dict, f)
    
    with open(cfg['label_freq_path'], 'w') as f:
        json.dump(label_freq_dict, f)


    # kmean_output_paths = []
    # for id in range(0, cfg['K']):
    #     idx_list = []
    #     for idx, path in enumerate(image_paths):
    #         if kmean_output[idx] == id:
    #             idx_list.append(idx)
    #     cluster_paths = [image_paths[idx] for idx in idx_list]
    #     kmean_output_paths.append(list(cluster_paths))

    # gen_cluster_result(dst_folder='data/kmean_output', clustering_output=kmean_output_paths)