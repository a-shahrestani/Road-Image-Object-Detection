import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import json


# a function to extract the labels and aggregate the from the json files
def _extract_json_labels(dict_address, write_flag=False, write_address=None):
    files = os.listdir(dict_address)
    labels = []
    for file in files:
        if file.endswith('.json'):
            data = pd.read_json(Path(dict_address) / file)
            objects = data.objects.tolist()
            for i in objects:
                labels.append(i['label'])
    labels = list(set(labels))
    if write_flag:
        with open(write_address + 'labels.txt', 'w') as fp:
            for item in labels:
                # write each item on a new line
                fp.write("%s\n" % item)
            print('Done')
    return labels


# a function to give the label indexing required for YOLO files
def _index_labels(labels):
    return {label: labels.index(label) for label in labels}


def _json_to_yolo(dict_address, labels_dict, verbose=0):
    files = os.listdir(dict_address)
    yolo_file = []
    for file in files:
        if file.endswith('.json'):
            with open(Path(dict_address) / file, 'r') as f:
                data = json.load(f)
            yolo_file = []
            img_width = data['width']
            img_height = data['height']
            objects = data['objects']
            for obj in objects:
                label_class = labels_dict[obj['label']]
                x = obj['bbox']['xmin'] / img_width
                y = obj['bbox']['ymin'] / img_height
                width = (obj['bbox']['xmax'] - obj['bbox']['xmin']) / img_width
                height = (obj['bbox']['ymax'] - obj['bbox']['ymin']) / img_height
                yolo_file.append(f'{label_class} {x} {y} {width} {height}')
            with open(Path(dict_address) / (file[:-5] + '.txt'), 'w') as fp:
                for item in yolo_file:
                    # write each item on a new line
                    fp.write("%s\n" % item)
            if verbose > 0:
                print(f'file {Path(dict_address) / file[:-5]} created')


# creating the config YAML file for YOLOv5 object detection model
def config_yaml(address, labels_dict, name='config'):
    classes = labels_dict.keys()
    yaml_info = {}
    yaml_info['path'] = ""
    yaml_info['train'] = ""
    yaml_info['val'] = ""
    yaml_info['nc'] = len(classes)
    yaml_info['names'] = {v: k for k, v in labels_dict.items()}
    print(yaml_info)
    with open(Path(address)/(name + '.yaml'), 'w') as f:
        yaml.dump(yaml_info,f,sort_keys=False)


if __name__ == '__main__':
    dict_address = '../../Datasets/Mapillary Traffic Sign Dataset/Fully Annotated/'
    # labels = _extract_json_labels(dict_address, write_flag=True, write_address=dict_address)
    # labels_dict = _index_labels(labels)
    # json_address = '../../Datasets/Mapillary Traffic Sign Dataset/Fully Annotated/classes.json'
    # json_file = json.dumps(labels_dict, indent=4)
    # with open('../../Datasets/Mapillary Traffic Sign Dataset/Fully Annotated/classes.json', 'w') as outfile:
    #     outfile.write(json_file)
    # _json_to_yolo(dict_address=dict_address, lebels_dict=labels_dict)
    with open(Path('../../Datasets/Mapillary Traffic Sign Dataset/Fully Annotated/') / 'classes.json', 'r') as f:
        data = json.load(f)
    config_yaml(dict_address,labels_dict=data)

