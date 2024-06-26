import pandas as pd
import cv2
import numpy as np
import os
from pillow_heif import register_heif_opener
from PIL import Image
import torchvision.models as models
from torch import nn
import torch

register_heif_opener()


def image_splitter(img, length, height):
    arr = img[:(img.shape[0] // height) * height, :(img.shape[1] // length) * length]
    t = np.array(np.vsplit(arr, arr.shape[0] // height))
    pieces = []
    for column in t:
        pieces += np.hsplit(column, column.shape[1] // length)
    pieces = np.array(pieces)
    return arr.shape, pieces


def reconstruct_image(pieces, shapes, length, height):
    img = np.zeros(shapes)
    counter = 0
    for i in range(shapes[0] // height):
        for j in range(shapes[1] // length):
            img[i * height:(i + 1) * height, j * length: (j + 1) * length] = pieces[counter]
            counter += 1
    cv2.imwrite('results/reconstructed.jpg', img)
    return img


def detect_cracks_fn(input_images, model_path, num_classes=4, num_shown_classes=2, classes_name=None, device='cpu',
                     image_length=256, image_height=256):
    # we will use the model to detect the cracks in the images
    resnet_model = models.resnet18(pretrained=False)
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
    resnet_model.load_state_dict(torch.load(model_path))
    resnet_model.eval()
    with (torch.no_grad()):
        images = torch.stack([tensor for tensor in input_images], dim=0).to(device)
        outputs = resnet_model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probs, dim=1)
        predicted_classes = predicted_classes.to('cpu').numpy()
        visualized_classes = [0 if class_code == classes_name.index('no_crack') else 1 for class_code in
                              predicted_classes]
        predicted_images = []
        visualized_images = []
        for i in range(len(input_images)):
            predicted_images.append(
                np.full((image_length, image_height), predicted_classes[i] * 255 // len(classes_name)))
            visualized_images.append(
                np.full((image_length, image_height), visualized_classes[i] * 255 // len(classes_name)))
        return predicted_images, visualized_images


def large_scale_splitter(source_directory, length, height, num_samples=500, save_directory='results', format='jpg',
                         resize=False, resize_degree=4, detect_cracks=False, model_address=None, classes_name=None):
    # we will use the image_splitter function to split the image into 256x256 pieces
    # we first have to read X number of images the images from the directory
    images = []
    image_names = os.listdir(source_directory)
    if num_samples > len(image_names):
        num_samples = len(image_names)
    if format != 'HEIC':
        for i in range(num_samples):
            images.append(cv2.imread(f'{source_directory}/{image_names[i]}',
                                     cv2.IMREAD_GRAYSCALE))
    else:
        for i in range(num_samples):
            images.append(np.array(Image.open(f'{source_directory}/{image_names[i]}').convert('L'))
                          )

    for i, image in enumerate(images):
        if resize:
            image = cv2.resize(image, (image.shape[1] // resize_degree, image.shape[0] // resize_degree))
        shapes, pieces = image_splitter(image, length, height)
        image_name = image_names[i].split('.')[0]
        for i, split in enumerate(pieces):
            cv2.imwrite(f'{save_directory}/{image_name}_piece{i}.jpg', split)
        img = reconstruct_image(pieces, shapes, length, height)
        cv2.imwrite(f'{save_directory}/{image_name}_reconstructed.jpg', img)
        if detect_cracks:
            # we will use the model to detect the cracks in the images
            predicted_images, visualized_images = detect_cracks_fn(pieces, model_address, num_classes=4,
                                                                   num_shown_classes=2, classes_name=classes_name,
                                                                   device='cpu', image_length=length,
                                                                   image_height=height)
            img = reconstruct_image(visualized_images, shapes, length, height)
            cv2.imwrite(f'{save_directory}/{image_name}_cracks_detected_visualized_reconstructed.jpg', img)
            img = reconstruct_image(predicted_images, shapes, length, height)
            cv2.imwrite(f'{save_directory}/{image_name}_cracks_detected_reconstructed.jpg', img)

    # sample = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)
    # shapes, pieces = image_splitter(sample, 256, 256)
    # for i, split in enumerate(pieces):
    #     cv2.imwrite(f'results/piece{i}.jpg', split)
    # img = reconstruct_image(pieces, shapes, 256, 256)

run_address = './output/classification/ResNet18/augmented_CWGAN/test2'
model_path = f'{run_address}/resnet_model_final.pth'
classes_name = ['alligator', 'longitudinal', 'transverse', 'no_crack']
large_scale_splitter('input/Additional', 256, 256, num_samples=20,
                     save_directory='results/Additional', format='HEIC',
                     resize=True, resize_degree=2, detect_cracks=True, model_address=model_path,
                     classes_name=classes_name)
