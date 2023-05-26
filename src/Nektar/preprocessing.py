import json
import os
import random

import pandas
import numpy


def only_sign_annotations():
    data_path = '../../../datasets/Yellowstone ATS/Annotations/trackb3.json'
    output = '../../../datasets/Yellowstone ATS/Annotations/trackb3_signs.json'
    data = None
    with open(data_path, 'r') as f:
        data = json.load(f)
        print(len(data['annotations']))
        data['annotations'] = [annotation for annotation in data['annotations'] if
                               not 6 < annotation['category_id'] < 40]
        print(len(data['annotations']))
    with open(output, 'w') as out_F:
        json.dump(data, out_F)


files = ['tracka1_signs', 'trackb1_signs', 'trackb3_signs', 'trackc1_signs']


def combine_tracks(address, files=[]):
    last_image_id = 0
    last_annotation_id = 0
    data_jsons = []
    for i, file in enumerate(files):
        with open(address + file + '.json', 'r') as f:
            data = json.load(f)
            for j, image in enumerate(data['images']):
                data['images'][j]['id'] += last_image_id
            for j, image in enumerate(data['annotations']):
                data['annotations'][j]['id'] += last_annotation_id
                data['annotations'][j]['image_id'] += last_image_id
            last_image_id += data['images'][-1]['id']
            last_annotation_id += data['annotations'][-1]['id']
            data_jsons.append(data)
    new_data = data_jsons[0]
    for data in data_jsons[1:]:
        new_data['images'] += data['images']
        new_data['annotations'] += data['annotations']
    with open(address + '/combined_annotations.json', 'w') as f:
        json.dump(new_data, f)


def train_test_split(json_file='', images='', percentage=0.1):
    with open(json_file, 'r') as f:
        data = json.load(f)
    # random.shuffle(data['images'])
    # test = data['images'][int(len(data['images']) * (1 - percentage)):]
    # train = data['images'][:int(len(data['images']) * (1 - percentage))]
    test = random.sample(data['images'], int(len(data['images']) * percentage))
    train = [image for image in data['images'] if image not in test]
    train_images = [image['file_name'] for image in train]
    test_images = [image['file_name'] for image in test]
    for image in train_images:
        os.rename(images + '/' + image, images + '/train/' + image)
    for image in test_images:
        os.rename(images + '/' + image, images + '/test/' + image)
    train_data = {'images': train, 'annotations': [], 'categories': data['categories'], 'licenses': data['licenses'],
                  'info': data['info']}
    test_data = {'images': test, 'annotations': [], 'categories': data['categories'], 'licenses': data['licenses'],
                  'info': data['info']}
    for annotation in data['annotations']:
        for image in train:
            if annotation['image_id'] == image['id']:
                train_data['annotations'].append(annotation)
        for image in test:
            if annotation['image_id'] == image['id']:
                test_data['annotations'].append(annotation)
    with open(json_file.replace('.json', '_train.json'), 'w') as f:
        json.dump(train_data, f)
    with open(json_file.replace('.json', '_test.json'), 'w') as f:
        json.dump(test_data, f)


# combine_tracks('../../../datasets/Yellowstone ATS/Annotations/',files)
# only_sign_annotations()
train_test_split(json_file='../../../datasets/Yellowstone ATS/Annotations/combined_annotations.json',
                 images='../../../datasets/Yellowstone ATS/Images')
