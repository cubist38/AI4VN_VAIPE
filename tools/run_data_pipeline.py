from typing import Dict
from clustering.run import do_clustering
import argparse
import json

parser = argparse.ArgumentParser(description='Data pipeline')
parser.add_argument('--image_src', default=None, type=str, help='Path to image folder')
parser.add_argument('--data_json', default=None, type=str, help='Path to data json file')
parser.add_argument('--save', default='data/cluster.json', type=str, help='Path to save new data json')
args = parser.parse_args()

def data_pipeline(image_src: str, image_dict: Dict) -> Dict:
    '''
        Args:
            image_src: path to image folder
            image_dict: a dictionary, format {
                image_1: label_of_image_1,
                image_2: label_of_image_2,
            }
    '''
    image_list = [image_name for image_name in image_dict.keys()]
    print('Running clustering ...')
    image_clusters = do_clustering(image_src, image_list)
    '''
        image_clusters = N x [] 
        where 
            - N is the number of clusters
            - each group contains a list of image path belong to that group
    '''
    print(f'Found {len(image_clusters)} clusters.')
    data_dict = {}
    for cluster_id, cluster in enumerate(image_clusters):
        freq = [0 for i in range(108)]
        for image_name in cluster:
            if image_dict[image_name] != 107:
                freq[image_dict[image_name]] += 1
        data_dict[cluster_id] = {}
        data_dict[cluster_id]['list'] = cluster
        data_dict[cluster_id]['freq'] = freq
    
    return data_dict

if __name__ == '__main__':
    with open(args.data_json) as f:
        image_dict = json.load(f)

    data_dict = data_pipeline(args.image_src, image_dict)
    with open(args.save, 'w') as f:
        json.dump(data_dict, f)
    print(f'Save to {args.save} successfully!')
