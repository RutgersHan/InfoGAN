from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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
            height, width, depth, embedding_num, \
                images, embeddings, labels = pickle.load(f)
            self._images = np.array([image for image in images])
            self._labels = np.array([label for label in labels])
            self._embeddings = np.array([
                embedding for embedding in embeddings])
            self._epochs_completed = -1
            self._num_examples = len(images)
            # shuffle on first run
            self._index_in_epoch = self._num_examples
            self._flip_flag = flip_flag
            self.train = self
            self.image_dim = height * width * depth
            self.image_shape = (height, width, depth)
            self.embedding_num = embedding_num
