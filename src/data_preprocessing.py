import os
import shutil
import numpy as np
import pandas as pd
import json


def get_file_names(address):
    with open(address, "r") as f:
        names = f.readlines()
        names = [name.strip() for name in names]
    return names


def get_dir_file_names(address):
    file_names = os.listdir(address)
    return file_names


def separate_labels(data_dir, target_dir, names, format='.txt'):
    non_existent = []
    if names or target_dir is not None:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for name in names:
            if os.path.exists(os.path.join(data_dir, name + format)):
                shutil.copy(os.path.join(data_dir, name + format), os.path.join(target_dir, name + format))
            else:
                print(f'file with name {name} does not exit')
                non_existent.append(name)
    print('Split is done!')
    return non_existent


def separate_data(img_dir, label_dir, label_train_dir, label_test_dir, img_train_dir, img_test_dir, label_format='.txt',
                  img_format='.jpg', train_test_split_percentage=30):
    non_existent = []
    if not os.path.exists(img_test_dir):
        os.makedirs(img_test_dir)
    if not os.path.exists(img_train_dir):
        os.makedirs(img_train_dir)
    if not os.path.exists(label_train_dir):
        os.makedirs(label_train_dir)
    if not os.path.exists(label_test_dir):
        os.makedirs(label_test_dir)
    label_files = [name[:-len(label_format)] for name in os.listdir(label_dir)]
    test_index = len(label_files) // 100 * (100-train_test_split_percentage)
    separate_labels(data_dir=label_dir, target_dir=label_train_dir, names=label_files[:test_index], format='.txt')
    separate_labels(data_dir=label_dir, target_dir=label_test_dir, names=label_files[test_index:], format='.txt')
    for file in label_files[:test_index]:
        if os.path.exists(os.path.join(img_dir, file + img_format)):
            shutil.copy(os.path.join(img_dir, file + img_format), os.path.join(img_train_dir, file + img_format))
        else:
            print(f'file with name {file} does not exit')
            non_existent.append(file)
    for file in label_files[test_index:]:
        if os.path.exists(os.path.join(img_dir, file + img_format)):
            shutil.copy(os.path.join(img_dir, file + img_format), os.path.join(img_test_dir, file + img_format))
        else:
            print(f'file with name {file} does not exit')
            non_existent.append(file)
    print('Data split is done!')
    return non_existent


if __name__ == '__main__':
    # file_names = []
    # non_existents = []
    # file_names.append(get_file_names('../../datasets/Mapillary Traffic Sign Dataset/Fully Annotated/splits/train.txt'))
    # file_names.append(get_file_names('../../datasets/Mapillary Traffic Sign Dataset/Fully Annotated/splits/test.txt'))
    # file_names.append(get_file_names('../../datasets/Mapillary Traffic Sign Dataset/Fully Annotated/splits/val.txt'))
    # target_dirs = ['training', 'testing', 'validation']
    # for i, names in enumerate(file_names):
    #     non_existents.append(
    #         separate_labels(data_dir='../../Datasets/Mapillary Traffic Sign Dataset/Fully Annotated/annotations/',
    #                         target_dir=
    #                         f'../../Datasets/Mapillary Traffic Sign Dataset/Fully Annotated/labels/{target_dirs[i]}/',
    #                         names=names))
    # non_existents = separate_data(img_dir='../../Datasets/Traffic Signs Dataset (Mapillary and DFG)/images/',
    #                               label_dir='../../Datasets/Traffic Signs Dataset (Mapillary and DFG)/txts (YOLO)/',
    #                               label_format='.txt',
    #                               img_format='.jpg',
    #                               img_train_dir='../../Datasets/Traffic Signs Dataset (Mapillary and DFG)/images/training/',
    #                               img_test_dir='../../Datasets/Traffic Signs Dataset (Mapillary and DFG)/images/testing/',
    #                               label_train_dir='../../Datasets/Traffic Signs Dataset (Mapillary and DFG)/labels/training/',
    #                               label_test_dir='../../Datasets/Traffic Signs Dataset (Mapillary and DFG)/labels/testing/',
    #                               train_test_split_percentage=10)
    # print(non_existents)
    address = '../../Datasets/Mapillary Traffic Sign Dataset/Fully Annotated/classes.json'
    with open(address, 'r') as f:
        data = json.load(f)
    print(data)

