from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import numpy as np
import tensorflow as tf
sys.path.append('../infogan/')
#from misc.datasets_embedding import FlowerDataset


import numpy as np
import pickle
import random


class Dataset(object):
    def __init__(self, images, embeddings, labels=None, flip_flag=True):
        self._images = images
        self._labels = labels
        self._embeddings = embeddings
        self._epochs_completed = -1
        self._num_examples = len(images)
        # shuffle on first run
        self._index_in_epoch = self._num_examples
        self._flip_flag = flip_flag

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def embeddings(self):
        return self._embeddings

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def transform(self, images):
        images = images.astype(np.float32) * (2. / 255) - 1.
        transformed_images = np.zeros_like(images)
        for i in range(images.shape[0]):
            if self._flip_flag:
                if random.random() > 0.5:
                    transformed_images[i] = np.fliplr(images[i])
                else:
                    transformed_images[i] = images[i]
        return transformed_images

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._embeddings = self._embeddings[perm]
            if self._labels is not None:
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        transformed_images = self.transform(self._images[start:end])
        if self._labels is None:
            return transformed_images, self._embeddings[start:end], None
        else:
            return (transformed_images, self._embeddings[start:end],
                    self._labels[start:end])


class FlowerDataset(Dataset):
    def __init__(self, pickle_path):
        self._pickle_path = pickle_path

    def get_data(self, flip_flag=True):
        with open(self._pickle_path, 'rb') as f:
            _, _, _, _, images, embeddings, labels = pickle.load(f)
            self._images = np.array([image for image in images])
            self._labels = np.array([label for label in labels])
            self._embeddings = np.array([
                embedding for embedding in embeddings])
            self._epochs_completed = -1
            self._num_examples = len(images)
            # shuffle on first run
            self._index_in_epoch = self._num_examples
            self._flip_flag = flip_flag




aaa = np.array([i for i in [1,2,3]])
file_name = '/home/han/Documents/CVPR2017/data/flowers/flowers64.pickle'


f_Dataset = FlowerDataset(file_name)
f_Dataset.get_data()


print(f_Dataset._num_examples)
images = f_Dataset._images
labels = f_Dataset._labels
embeddings = f_Dataset._embeddings


for i in range(2):
    a,b,c = f_Dataset.next_batch(3)

print('sss')
