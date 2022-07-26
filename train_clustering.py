from clustering.run import do_clustering
import yaml
import json

if __name__ == '__main__':
    cfg = yaml.safe_load(open('configs/config_clustering.yaml'))
    
    (image_paths, kmean_output), detection_data_dict, label_freq_dict = do_clustering(cfg)
    
    with open(cfg['detection_data_path'], 'w') as f:
        json.dump(detection_data_dict, f)
    
    with open(cfg['label_freq_path'], 'w') as f:
        json.dump(label_freq_dict, f)

    # Generate image folders of the above result:
    # gen_cluster_result(dst_folder='data/kmean_output', image_paths=image_paths, labels=kmean_output)