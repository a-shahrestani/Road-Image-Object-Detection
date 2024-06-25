import pandas as pd
import cv2
import numpy as np
import os
from pillow_heif import register_heif_opener
from PIL import Image

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


def large_scale_splitter(source_directory, length, height, num_samples=500, save_directory='results', format='jpg',
                         resize=False, resize_degree=4):
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


# sample = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)
# shapes, pieces = image_splitter(sample, 256, 256)
# for i, split in enumerate(pieces):
#     cv2.imwrite(f'results/piece{i}.jpg', split)
# img = reconstruct_image(pieces, shapes, 256, 256)


large_scale_splitter('input/Additional', 256, 256, num_samples=20, save_directory='results/Additional', format='HEIC',
                     resize=True, resize_degree=2)
