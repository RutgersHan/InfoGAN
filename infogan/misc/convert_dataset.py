from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np
import h5py
import os
import pickle
from utils import get_image, colorize
# from glob import glob

# TODO: 1. current label is temporary, need to change according to real label
#       2. Current, only split the data into train, need to handel train, test

IMSIZE = 64
FLOWER_DIR = '/home/han/Documents/CVPR2017/data/flowers'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def convert_flowers_dataset_pickle(data_dir, train_ratio=0.75):
    h = h5py.File(os.path.join(data_dir, 'flower_tv.hdf5'))
    outfile = os.path.join(data_dir, 'flowers' + str(IMSIZE)
                                               + '.pickle')
    flower_captions = {}
    for ds in h.iteritems():
        flower_captions[ds[0]] = np.array(ds[1])
        image_list = [key for key in flower_captions]
    image_list.sort()
    training_num = int(len(image_list) * train_ratio)
    training_image_list = image_list[0:training_num]
    height = IMSIZE
    width = IMSIZE
    depth = 3
    embedding_num = 5
    images = []
    embeddings = []
    labels = []
    for i, f in enumerate(training_image_list):
        f_name = os.path.join(data_dir, 'jpg', f)
        image = get_image(f_name, IMSIZE, is_crop=False, resize_w=IMSIZE)
        image = colorize(image)
        assert image.shape == (IMSIZE, IMSIZE, 3)
        image += 1.
        image *= (255. / 2.)
        image = image.astype('uint8')
        embedding = flower_captions[f]
        label = 0  # temporary
        print('%d\t%d' % (i, label))
        images.append(image)
        embeddings.append(embedding)
        labels.append(label)
    with open(outfile, 'wb') as f_out:
        pickle.dump([height, width, depth, embedding_num, images,
                     embeddings, labels], f_out)


def convert_flowers_dataset_tf(data_dir, train_ratio=0.75):
    h = h5py.File(os.path.join(data_dir, 'flower_tv.hdf5'))
    outfile = os.path.join(data_dir, 'flowers' + str(IMSIZE)
                                               + '.tfrecords')
    flower_captions = {}
    for ds in h.iteritems():
        flower_captions[ds[0]] = np.array(ds[1])
        image_list = [key for key in flower_captions]
    image_list.sort()
    training_num = int(len(image_list) * train_ratio)
    training_image_list = image_list[0:training_num]
    writer = tf.python_io.TFRecordWriter(outfile)
    for i, f in enumerate(training_image_list):
        f_name = os.path.join(data_dir, 'jpg', f)
        image = get_image(f_name, IMSIZE, is_crop=False, resize_w=IMSIZE)
        image = colorize(image)
        assert image.shape == (IMSIZE, IMSIZE, 3)
        image += 1.
        image *= (255. / 2.)
        image = image.astype('uint8')

        image_raw = image.tostring()
        embedding = flower_captions[f]
        label = 0  # temporary
        print('%d\t%d' % (i, label))
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(IMSIZE),
            'width': _int64_feature(IMSIZE),
            'depth': _int64_feature(3),
            'embedding_num': _int64_feature(5),
            'image_raw': _bytes_feature(image_raw),
            'embedding': _float_feature(embedding.astype(
                'float32').flatten().tolist()),
            'label': _int64_feature(label)
            }))
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    convert_flowers_dataset_pickle(FLOWER_DIR)
    # convert_flowers_dataset_tf(FLOWER_DIR)
