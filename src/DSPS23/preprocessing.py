import glob
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


def class_split(images_path='', class_files_path=''):
    class_files = os.listdir(class_files_path)
    for file in class_files:
        with open(class_files_path + '/' + file, 'r') as f:
            class_list = f.read().splitlines()
        for image in class_list:
            os.rename(images_path + image,
                      images_path + file.replace('.txt','') + '/' + image)


def stratified_train_test_split(json_file='', images='', percentage=0.1, class_files=[]):
    with open(json_file, 'r') as f:
        data = json.load(f)
    classes_list = []
    for file in class_files:
        # read data with names from text files containing names of images in each class and add them to classes
        with open(file, 'r') as f:
            classes_list.append(f.read().splitlines())

    with open(json_file, 'r') as f:
        data = json.load(f)
    # random.shuffle(data['images'])
    # test = data['images'][int(len(data['images']) * (1 - percentage)):]
    # train = data['images'][:int(len(data['images']) * (1 - percentage))]
    test = []
    train = []
    for class_list in classes_list:
        test.append(random.sample(class_list, int(len(class_list) * percentage)))
        train.append([image for image in class_list if image not in test[-1]])
    train = [item for sublist in train for item in sublist]
    test = [item for sublist in test for item in sublist]
    for image in train:
        os.rename(images + '/' + image, images + '/train/' + image)
    for image in test:
        os.rename(images + '/' + image, images + '/test/' + image)
    train_data = {'images': [], 'annotations': [], 'categories': data['categories'], 'licenses': data['licenses'],
                  'info': data['info']}
    test_data = {'images': [], 'annotations': [], 'categories': data['categories'], 'licenses': data['licenses'],
                 'info': data['info']}

    for image in data['images']:
        if image['file_name'] in train:
            image_id = image['id']
            train_data['images'].append(image)
            for annotation in data['annotations']:
                if annotation['image_id'] == image_id:
                    train_data['annotations'].append(annotation)
        elif image['file_name'] in test:
            image_id = image['id']
            test_data['images'].append(image)
            for annotation in data['annotations']:
                if annotation['image_id'] == image_id:
                    test_data['annotations'].append(annotation)

    with open(json_file.replace('.json', '_train.json'), 'w') as f:
        json.dump(train_data, f)
    with open(json_file.replace('.json', '_test.json'), 'w') as f:
        json.dump(test_data, f)


if __name__ == '__main__':
    # combine_tracks('../../../datasets/Yellowstone ATS/Annotations/',files)
    # only_sign_annotations()
    # files = ['instances_default_td4', 'instances_default_td5', 'instances_default_tdc2', 'instances_default_ts1']
    # combine_tracks('../../../datasets/Pavement Crack Detection/DSPS23 Pavement/Task 1 Crack Type/training_data/Annotations/',files)
    # train_test_split(json_file='../../../datasets/Pavement Crack Detection/DSPS23 Pavement/Task 1 Crack Type/training_data/Annotations/combined_annotations.json',
    #                  images='../../../datasets/Pavement Crack Detection/DSPS23 Pavement/Task 1 Crack Type/training_data')

    stratified_train_test_split(
        json_file='../../../datasets/Pavement Crack Detection/Crack500-Forest Annotated/Annotations/instances_default.json',
        images='../../../datasets/Pavement Crack Detection/Crack500-Forest Annotated/Images',
        class_files=glob.glob('../../../datasets/Pavement Crack Detection/Crack500-Forest Annotated/Classes/*.txt'))
    # class_split(images_path='../../../datasets/Pavement Crack Detection/Crack500-Forest Annotated/Images/',
    #             class_files_path='../../../datasets/Pavement Crack Detection/Crack500-Forest Annotated/Classes/')
