from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pickle
import random


class Dataset(object):
    def __init__(self, images, masks, embeddings, labels=None, flip_flag=True):
        self._images = images
        self._masks = masks
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
        # if the input image has values in [0, 255], use this function
        images = images.astype(np.float32) * (2. / 255) - 1.
        transformed_images = np.zeros_like(images)
        for i in range(images.shape[0]):
            if self._flip_flag:
                if random.random() > 0.5:
                    transformed_images[i] = np.fliplr(images[i])
                else:
                    transformed_images[i] = images[i]
        return transformed_images

    def sample_embeddings(self, embeddings):
        if len(embeddings.shape) == 2 or embeddings.shape[1] == 1:
            return np.squeeze(embeddings)
        else:
            batch_size, embedding_num, _ = embeddings.shape
            randix = np.random.randint(embedding_num, size=batch_size)
            return np.squeeze(embeddings[np.arange(batch_size), randix, :])

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
            self._masks = self._masks[perm]
            self._embeddings = self._embeddings[perm]
            if self._labels is not None:
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        # if the input image has values in [0, 255], use this self.transform()
        # sampled_images = self.transform(self._images[start:end])
        sampled_images = self._images[start:end]
        sampled_masks = self._masks[start:end]
        sampled_embeddings = self.sample_embeddings(
            self._embeddings[start:end])
        if self._labels is None:
            return sampled_images, sampled_masks, sampled_embeddings, None
        else:
            return (sampled_images, sampled_masks, sampled_embeddings,
                    self._labels[start:end])


class VisualizeData(object):
    def __init__(self, workdir, dataset):
        with open('%s/sample_captions/caption_embedding_image_name_%s.pickle'
                  % (workdir, dataset), 'rb') as f:
            self.captions, self.embeddings, self.images, self.filenames = pickle.load(f)
        self.caption_num = len(self.captions)
        # Change images value from [0, 255] to [-1., 1.]
        self.images = self.images.astype(np.float32) / 127.5 - 1.
        for i in range(self.caption_num):
            s = self.filenames[i]
            self.filenames[i] = s[s.find('/') + 1:]


class FlowerDataset(object):
    def __init__(self, workdir):
        self.image_shape = [64, 64, 3]
        self.image_dim = 64 * 64 * 3
        self.embedding_shape = [4800]
        self.train = None
        self.test = None
        self.fixedvisual_train = VisualizeData(workdir, 'train')
        self.fixedvisual_test = VisualizeData(workdir, 'test')
        # print(self.fixedvisual_train.filenames)
        # print(self.fixedvisual_test.filenames)

    def get_data(self, pickle_path, flip_flag=True):
        with open(pickle_path, 'rb') as f:
            # images value in [-1, 1]
            height, width, depth, embedding_num, \
                images, masks, embeddings, labels = pickle.load(f)
            array_images = np.array([image for image in images])
            array_masks = np.array([mask for mask in masks])
            array_labels = np.array([label for label in labels])
            array_embeddings = np.array([
                embedding for embedding in embeddings])
            # print(type(masks), len(masks))
            # print(type(array_masks), array_masks.shape)
            # print(type(array_masks[0]), array_masks[0].shape)
            # return
            self.image_dim = height * width * depth
            self.image_shape = [height, width, depth]
            self.embedding_shape = [array_embeddings.shape[-1]]
            # self.train = Dataset(array_images, array_embeddings, array_labels)
            return Dataset(array_images, array_masks, array_embeddings, array_labels)


class VisualizeAttrData(object):
    def __init__(self, images, embeddings, captions, vis_num, dataset):
        # randix = np.random.randint(len(captions), size=vis_num)
        # print(randix)
        if dataset == 'test':
            randix = [1478, 2699, 2910, 400, 2624, 924, 1933, 533, 2044, 2910]
        else:
            randix = [8362, 6141, 3614, 7870, 8146, 3028, 5426, 5367, 7215, 8542,
                      67, 2464, 1141, 1393, 3879, 901, 1329, 507, 8651, 119]
        vis_num = len(randix)
        self.caption_num = vis_num
        self.images = []
        self.embeddings = []
        self.captions = []
        self.filenames = []
        for i in range(self.caption_num):
            self.images.append(images[randix[i]])
            self.embeddings.append(embeddings[randix[i]])
            s = captions[randix[i]][0]
            # print('image: ', i)
            # print(s)
            self.captions.append(s)  # (s.replace("\n", ";"))
            self.filenames.append(i)
        self.images = np.array(self.images)
        self.embeddings = np.array(self.embeddings)
        # print(self.images.shape, self.embeddings.shape)
        # print(self.captions)


class BirdAttribuetDataset(object):
    def __init__(self):
        self.image_shape = [64, 64, 3]
        self.image_dim = 64 * 64 * 3
        self.embedding_shape = [312]
        self.train = None
        self.test = None
        self.fixedvisual_train = None
        self.fixedvisual_test = None

    def get_data(self, pickle_path, vis_num, dataset, flip_flag=True):
        with open(pickle_path, 'rb') as f:
            height, width, depth, embedding_num, \
                images, masks, embeddings, attributes, labels = pickle.load(f)

            array_images = np.array([image for image in images])
            array_masks = np.array([mask for mask in masks])
            array_labels = None  # np.array([label for label in labels])
            array_embeddings = np.array([
                embedding for embedding in embeddings])

            self.image_dim = height * width * depth
            self.image_shape = [height, width, depth]
            self.embedding_shape = [array_embeddings.shape[-1]]
            data = Dataset(array_images, array_masks, array_embeddings, array_labels)
            print('array_images', array_images.shape)
            print('array_masks', array_masks.shape)
            print('array_embeddings', array_embeddings.shape)
            # print(type(attributes), len(attributes), attributes[0])
            fixedvisual = VisualizeAttrData(images, embeddings, attributes, vis_num, dataset)

            return data, fixedvisual
