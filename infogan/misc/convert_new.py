from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import tensorflow as tf
import numpy as np
import h5py
import os
import pickle
from infogan.misc.utils import get_image, colorize
import scipy.misc
import scipy.io as sio
import pandas as pd

# from glob import glob

# TODO: 1. current label is temporary, need to change according to real label
#       2. Current, only split the data into train, need to handel train, test

IMSIZE = 128  # 64
LOAD_SIZE = 152  # 76
# BIRD_DIR = '/home/han/Documents/CVPR2017/InfoGAN/Data/birds'
BIRD_DIR = '/home/tao/deep_learning/CVPR2017/icml_origin/InfoGAN/Data/birds'


def load_mask(data_dir, bbox):
    if bbox:
        filenames_masks = sio.loadmat(os.path.join(data_dir, 'mask_cropped.mat'))
    else:
        filenames_masks = sio.loadmat(os.path.join(data_dir, 'mask.mat'))
    filenames = filenames_masks['filenames']
    masks = filenames_masks['masks']
    print(type(filenames[0][0]), type(filenames[0][0][0]), filenames[0][0][0])
    print(type(masks[0][0]), masks[0][0].shape)

    mask_dic = {}
    for i in range(len(filenames)):
        key = filenames[i][0][0]
        # print key
        data = masks[i][0]
        mask_dic[key] = data
    return mask_dic


def get_icml16_text_embedding(data_dir):
    with open(os.path.join(data_dir, 'icml16_text_embedding.p'), 'rb') as f:
        embeddings = pickle.load(f)
        print('text_embedding: ', len(embeddings))
    return embeddings


def get_class_image_list(data_dir, total_class_num=200):
    # <class_id> <class_name>
    training_class_list = pd.read_csv(os.path.join(data_dir, 'trainvalids.txt'),
                                      delim_whitespace=True, header=None)[0].tolist()
    test_class_list = []
    for i in range(1, total_class_num + 1):
        if i not in training_class_list:
            test_class_list.append(i)
    training_class_list.sort()
    test_class_list.sort()
    img_filenames = pd.read_csv(os.path.join(data_dir, 'images.txt'),
                                delim_whitespace=True, header=None)[1].tolist()
    count = 0
    class_filenames = {}
    for f_name in img_filenames:
        id = int(f_name.split('.')[0])
        if id not in class_filenames:
            class_filenames[id] = []
            class_filenames[id].append(f_name)
        else:
            class_filenames[id].append(f_name)
        count = count + 1
    print(count)
    return training_class_list, test_class_list, class_filenames, img_filenames


def convert_birds_dataset_attribute_pickle(data_dir):
    bbox = 0 # Warning: bbox=1 is not tested
    b_img = 1
    icml16_btext = 1

    training_class_list, test_class_list, \
        class_filenames, img_filenames = get_class_image_list(data_dir)

    if icml16_btext:
        icml16_text_captions = get_icml16_text_embedding(data_dir)
    else:
        icml16_text_captions = None

    save_data_list(training_class_list, class_filenames, data_dir, bbox, b_img,
                   icml16_text_captions,
                   os.path.join(data_dir, 'pickle_list/train/'))
    # ## For Test data
    save_data_list(test_class_list, class_filenames, data_dir, bbox, b_img,
                   icml16_text_captions,
                   os.path.join(data_dir, 'pickle_list/test/'))




def save_file_names(class_list, image_lists, outpath):
    filenames = []
    for i, c in enumerate(class_list):
        for j, f in enumerate(image_lists[c]):
            filenames.append(f.replace('.jpg', ''))
    if len(filenames):
        print('filenames', len(filenames), filenames[0])
        outfile = outpath + 'filenames.pickle'
        with open(outfile, 'wb') as f_out:
            pickle.dump(filenames, f_out)
            print('save to: ', outfile)


def save_data_list(class_list, image_lists, data_dir, bbox, b_img,
                   icml16_text_captions, outpath):
    images = []
    icml16_text_embeddings = []
    class_id = []
    filenames = []
    class_range = {}
    count = 0
    if not os.path.exists(outpath):
        os.makedirs(outpath)


    if bbox:
        image_dir = os.path.join(data_dir, 'images_cropped')
    else:
        image_dir = os.path.join(data_dir, 'images')

    for i, c in enumerate(class_list):
        class_range[c] = []
        class_range[c].append(count)
        for j, f in enumerate(image_lists[c]):
            filenames.append(f.replace('.jpg', ''))
            class_id.append(c)
            if b_img:
                f_name = os.path.join(image_dir, f)
                if bbox:
                    img = get_image(f_name, IMSIZE, is_crop=False, resize_w=IMSIZE)
                else:
                    img = get_image(f_name, LOAD_SIZE, is_crop=False, resize_w=LOAD_SIZE)
                images.append(colorize(img))

            if icml16_text_captions is not None:
                icml16_text_embeddings.append(icml16_text_captions[f])
            count = count + 1
            print(count)
        class_range[c].append(count - 1)

    if len(images):
        print('images', len(images), images[0].shape)
        if bbox:
            outfile = outpath + str(IMSIZE) + 'cropped_images.pickle'
        else:
            outfile = outpath + str(LOAD_SIZE) + 'images.pickle'
        with open(outfile, 'wb') as f_out:
            pickle.dump(images, f_out)
            print('save to: ', outfile)
    if len(icml16_text_embeddings):
        print('icml16_text_embeddings', len(icml16_text_embeddings),
              icml16_text_embeddings[0].shape)
        outfile = outpath + 'icml16_text_embeddings.pickle'
        with open(outfile, 'wb') as f_out:
            pickle.dump(icml16_text_embeddings, f_out)
            print('save to: ', outfile)
    if len(filenames):
        print('filenames', len(filenames), filenames[0])
        outfile = outpath + 'filenames.pickle'
        with open(outfile, 'wb') as f_out:
            pickle.dump(filenames, f_out)
            print('save to: ', outfile)
    if len(class_id):
        outfile = outpath + 'class_info.pickle'
        with open(outfile, 'wb') as f_out:
            pickle.dump([class_id, class_range], f_out)
            print('save to: ', outfile)


if __name__ == '__main__':
    # get_class_image_list(BIRD_DIR)
    convert_birds_dataset_attribute_pickle(BIRD_DIR)
