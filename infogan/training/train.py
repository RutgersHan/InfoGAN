from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np


def read_and_decode_with_labels(filename_queue, flip_flag=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'embedding_num': tf.FixedLenFeature([], tf.int64),
            'image_raw': _bytes_feature(image_raw),
            'embedding': tf.FixedLenFeature([], tf.float32),
            'label': tf.FixedLenFeature([], tf.int64),

        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape(features['height'] * features['width'] * features['depth'])
    image = tf.reshape(image, [features['height'], features['width'],
                               features['depth']])
    image = tf.cast(image, tf.float32) * (2. / 255) - 1.
    if flip_flag:
        image = tf.image.random_flip_left_right(image)
    embedding = tf.reshape(features[embedding], [features[embeding_num], -1])
    label = tf.cast(features['label'], tf.int32)

return image, embedding, label
