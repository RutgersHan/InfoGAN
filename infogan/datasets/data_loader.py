from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import h5py
import os
import random
from PIL import Image
import skimage
import skimage.io
import skimage.transform


def read_img(img_path, img_size, flip_flag=True):
    im = Image.open(img_path)
    im_resized = im.resize(img_size)
    im_resized = np.array(im_resized)
    if flip_flag:
        if random.random() > 0.5:
            im_resized = np.fliplr(im_resized)
    return im_resized.astype('float32')


def load_training_data(data_dir, data_set):
    if data_set == 'flowers':
        training_radio = 0.75
        h = h5py.File(os.path.join(data_dir, 'flower_tv.hdf5'))
        flower_captions = {}
        for ds in h.iteritems():
            flower_captions[ds[0]] = np.array(ds[1])
            image_list = [key for key in flower_captions]
        image_list.sort()
        training_num = int(len(image_list) * training_radio)
        training_image_list = image_list[0:training_num]
        test_image_list = image_list[training_num:]
        return {
            'image_list': training_image_list,
            'captions': flower_captions,
            'data_length': len(training_image_list)
            }

    else:
        raise NotImplementedError


file_path = '/home/han/Documents/CVPR2017/text-to-image/Data'
jpg_path = 'flowers/jpg/image_00006.jpg'
my_dict = load_training_data(file_path, 'flowers')
current_file = os.path.join(file_path, jpg_path)
img = read_img(current_file,[64,64], False)
img_0 = img/127.5 -1
img1 = skimage.io.imread(current_file)
img_resized = skimage.transform.resize(img1, (64, 64))
img_resized_0 = img_resized - 1
skimage.io.imshow(img_resized)
