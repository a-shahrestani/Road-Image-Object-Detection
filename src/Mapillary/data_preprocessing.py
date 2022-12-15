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
    test_index = len(label_files) // 100 * (100 - train_test_split_percentage)
    separate_labels(data_dir=label_dir, target_dir=label_train_dir, names=label_files[:test_index], format='.txt')
    separate_labels(data_dir=label_dir, target_dir=label_test_dir, names=label_files[test_index:], format='.txt')
    # for file in label_files[:test_index]:
    #     if os.path.exists(os.path.join(img_dir, file + img_format)):
    #         shutil.copy(os.path.join(img_dir, file + img_format), os.path.join(img_train_dir, file + img_format))
    #     else:
    #         print(f'file with name {file} does not exit')
    #         non_existent.append(file)
    # for file in label_files[test_index:]:
    #     if os.path.exists(os.path.join(img_dir, file + img_format)):
    #         shutil.copy(os.path.join(img_dir, file + img_format), os.path.join(img_test_dir, file + img_format))
    #     else:
    #         print(f'file with name {file} does not exit')
    #         non_existent.append(file)
    print('Data split is done!')
    return non_existent


def get_merged_classes(address, pattern='--g'):
    with open(address, 'r') as f:
        data = json.load(f)
    classes = list(data.keys())
    merged_classes = list(set([class_name[:class_name.find(pattern)] for class_name in classes]))
    merged_classes_dict = {merged_class: merged_classes.index(merged_class) for merged_class in merged_classes}
    return data, merged_classes_dict


def merge_image_classes(classes_address, data_dir, target_dir, save_classes=True, merged_classes_address=None,
                        pattern='--g'):
    classes_dict, merged_classes_dict = get_merged_classes(address=classes_address, pattern=pattern)
    if save_classes:
        with open(merged_classes_address, 'w') as f:
            json.dump(merged_classes_dict, f)
    file_names = get_dir_file_names(address=data_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for file_name in file_names:
        with open(os.path.join(data_dir, file_name), 'r') as f:
            lines = f.readlines()
        with open(os.path.join(target_dir, file_name), 'w') as f:
            for line in lines:
                class_code = int(line.split()[0])
                if class_code in classes_dict.values():
                    class_name = list(classes_dict.keys())[list(classes_dict.values()).index(class_code)]
                    class_name = class_name[:class_name.find(pattern)]
                    merged_class_code = merged_classes_dict[class_name]
                    f.write(
                        f'{merged_class_code} {line.split()[1]} {line.split()[2]} {line.split()[3]} {line.split()[4]}' + '\n')
                else:
                    print(f'class {class_code} does not exist in the classes dictionary')


def get_classes_cvat(classes_address, target_address):
    with open(classes_address, 'r') as f:
        data = json.load(f)
    classes = list(data.keys())
    classes_list = [{"name": class_name, "attributes": []} for class_name in classes]
    with open(target_address, 'w') as f:
        json.dump(classes_list, f)

    pass


def find_class_instances(classes_address, data_dir):
    with open(classes_address, 'r') as f:
        classes_dict = json.load(f)
    file_names = get_dir_file_names(address=data_dir)
    class_instances = {class_name: 0 for class_name in classes_dict.keys()}
    for file_name in file_names:
        with open(os.path.join(data_dir, file_name), 'r') as f:
            lines = f.readlines()
        for line in lines:
            class_code = int(line.split()[0])
            if class_code in classes_dict.values():
                class_name = list(classes_dict.keys())[list(classes_dict.values()).index(class_code)]
                class_instances[class_name] += 1
    return class_instances


def remove_classes(classes_address, new_classes_address, data_dir, target_dir, class_names):
    with open(classes_address, 'r') as f:
        classes_dict = json.load(f)
    class_codes = [int(classes_dict[class_name]) for class_name in class_names]
    for name in class_names:
        classes_dict.pop(name, None)
    new_classes_dict = {class_name: i for i, class_name in enumerate(classes_dict.keys())}
    with open(new_classes_address, 'w') as f:
        json.dump(new_classes_dict, f)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    file_names = get_dir_file_names(address=data_dir)
    empty_files_counter = 0
    for file_name in file_names:
        with open(os.path.join(data_dir, file_name), 'r') as f:
            lines = f.readlines()
        lines = [line for line in lines if int(line.split()[0]) not in class_codes]
        if len(lines) > 0:
            with open(os.path.join(target_dir, file_name), 'w') as f:
                for line in lines:
                    old_class_code = int(line.split()[0])
                    new_class_code = new_classes_dict[[k for k, v in classes_dict.items() if v == old_class_code][0]]
                    f.write(
                        f'{new_class_code} {line.split()[1]} {line.split()[2]} {line.split()[3]} {line.split()[4]}'
                        + '\n')
        else:
            empty_files_counter += 1
    print(f'{empty_files_counter} files are empty')


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
    # get_classes_cvat(classes_address='../../Datasets/Mapillary Traffic Sign Dataset/Fully Annotated/classes.json',
    #                  target_address='../../Datasets/Mapillary Traffic Sign Dataset/Fully Annotated/classes_cvat.json')
    address = '../../../datasets/Mapillary Traffic Sign Dataset/Fully Annotated/merged_classes.json'
    new_address = '../../../datasets/Mapillary Traffic Sign Dataset/Fully Annotated/reduced_merged_classes.json'
    # merged_classes_address = '../../Datasets/Mapillary Traffic Sign Dataset/Fully Annotated/merged_classes.json'
    data_dir = '../../../datasets/Mapillary Traffic Sign Dataset/Fully Annotated/labels/training_merged/'
    target_dir = '../../../datasets/Mapillary Traffic Sign Dataset/Fully Annotated/labels/training/'
    # target_dir = '../../Datasets/Mapillary Traffic Sign Dataset/Fully Annotated/labels/training_merged/'
    # merge_image_classes(classes_address=address, data_dir=data_dir, target_dir=target_dir,
    #                     merged_classes_address=merged_classes_address)
    # instances = find_class_instances(classes_address=address, data_dir=data_dir)
    remove_classes(classes_address=address, new_classes_address=new_address, data_dir=data_dir, target_dir=target_dir,
                   class_names=['other-sig'])
