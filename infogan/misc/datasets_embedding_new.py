from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import pickle
import random
import scipy.misc


class Dataset(object):
    def __init__(self, images, imsize, embeddings=None,
                 filenames=None, workdir=None,
                 labels=None, aug_flag=True,
                 class_id=None, class_range=None):
        self._images = images
        self._embeddings = embeddings
        self._filenames = filenames
        self.workdir = workdir
        self._labels = labels
        self._epochs_completed = -1
        self._num_examples = len(images)
        # shuffle on first run
        self._index_in_epoch = self._num_examples
        self._aug_flag = aug_flag
        self._class_id = np.array(class_id)
        self._class_range = class_range
        self._imsize = imsize
        self._perm = None

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
    def filenames(self):
        return self._filenames

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def transform(self, images):
        if self._aug_flag:
            transformed_images = np.zeros([images.shape[0], self._imsize, self._imsize, 3])
            ori_size = images.shape[1]
            for i in range(images.shape[0]):
                h1 = int(np.floor((ori_size - self._imsize) * np.random.random()))
                w1 = int(np.floor((ori_size - self._imsize) * np.random.random()))
                cropped_image = images[i][w1: w1 + self._imsize, h1: h1 + self._imsize, :]
                if random.random() > 0.5:
                    transformed_images[i] = np.fliplr(cropped_image)
                else:
                    transformed_images[i] = cropped_image
            return transformed_images
        else:
            return images

    def sample_embeddings(self, embeddings, filenames, sample_num):
        if len(embeddings.shape) == 2 or embeddings.shape[1] == 1:
            return np.squeeze(embeddings)
        else:
            batch_size, embedding_num, _ = embeddings.shape
            # Take every sample_num captions to compute the mean vector
            sampled_embeddings = []
            sampled_captions = []
            for i in range(batch_size):
                randix = np.random.choice(embedding_num, sample_num, replace=False)
                if sample_num == 1:
                    randix = int(randix)
                    with open(self.workdir + '/text_c10/' + filenames[i] + '.txt', "r") as f:
                        captions = f.read().split('\n')
                    captions = [cap for cap in captions if len(cap) > 0]
                    # print(captions)
                    sampled_captions.append(captions[randix])
                    sampled_embeddings.append(embeddings[i, randix, :])
                else:
                    e_sample = embeddings[i, randix, :]
                    e_mean = np.mean(e_sample, axis=0)
                    sampled_embeddings.append(e_mean)
            sampled_embeddings_array = np.array(sampled_embeddings)
            return np.squeeze(sampled_embeddings_array), sampled_captions

    def next_batch(self, batch_size, window):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            self._perm = np.arange(self._num_examples)
            np.random.shuffle(self._perm)

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        current_ids = self._perm[start:end]
        fake_ids = np.random.randint(self._num_examples, size=batch_size)
        collision_flag = (self._class_id[current_ids] == self._class_id[fake_ids])
        fake_ids[collision_flag] = (fake_ids[collision_flag] +
                                    np.random.randint(100, 200)) % self._num_examples

        sampled_images = self._images[current_ids]
        sampled_wrong_images = self._images[fake_ids, :, :, :]
        sampled_images = sampled_images.astype(np.float32)
        sampled_wrong_images = sampled_wrong_images.astype(np.float32)
        sampled_images = sampled_images * (2. / 255) - 1.
        sampled_wrong_images = sampled_wrong_images * (2. / 255) - 1.

        sampled_images = self.transform(sampled_images)
        sampled_wrong_images = self.transform(sampled_wrong_images)
        ret_list = [sampled_images, sampled_wrong_images]

        if self._embeddings is not None:
            filenames = [self._filenames[i] for i in current_ids]
            sampled_embeddings, sampled_captions = \
                self.sample_embeddings(self._embeddings[current_ids],
                                       filenames, window)
            ret_list.append(sampled_embeddings)
            ret_list.append(sampled_captions)
        else:
            ret_list.append(None)
            ret_list.append(None)

        if self._labels is not None:
            ret_list.append(self._labels[current_ids])
        else:
            ret_list.append(None)
        return ret_list

    def next_batch_test(self, batch_size, start):
        """Return the next `batch_size` examples from this data set."""
        if (start + batch_size) > self._num_examples:
            end = self._num_examples
            start = end - batch_size
        else:
            end = start + batch_size

        sampled_images = self._images[start:end]
        sampled_images = sampled_images.astype(np.float32)
        sampled_images = sampled_images * (2. / 255) - 1.  # from [0, 255] to [-1.0, 1.0]
        sampled_images = self.transform(sampled_images)

        sampled_filenames = self._filenames[start:end]
        sampled_embeddings = self._embeddings[start:end]
        _, embedding_num, _ = sampled_embeddings.shape
        sampled_embeddings_batchs = []
        for i in range(embedding_num):
            batch = sampled_embeddings[:, i, :]
            sampled_embeddings_batchs.append(np.squeeze(batch))

        return sampled_images, sampled_embeddings_batchs, sampled_filenames


class TextDataset(object):
    def __init__(self, workdir):
        self.image_shape = [64, 64, 3]
        self.image_dim = 64 * 64 * 3
        self.embedding_shape = [1024]
        self.train = None
        self.test = None
        self.workdir = workdir

    def get_data(self, pickle_path, aug_flag=True):
        with open(pickle_path + '/76images.pickle', 'rb') as f:
            images = pickle.load(f)
            array_images = np.array([image for image in images])
            print('array_images: ', array_images.shape)

        with open(pickle_path + '/icml16_text_embeddings.pickle', 'rb') as f:
            embeddings = pickle.load(f)
            array_embeddings = np.array([  # every 2400D has alreay been L2 Normalized
                embedding for embedding in embeddings])
            self.embedding_shape = [array_embeddings.shape[-1]]
            print('array_embeddings: ', array_embeddings.shape)
        with open(pickle_path + '/filenames.pickle', 'rb') as f:
            list_filenames = pickle.load(f)
            print('list_filenames: ', len(list_filenames), list_filenames[0])
        with open(pickle_path + '/class_info.pickle', 'rb') as f:
            class_id, class_range = pickle.load(f)  # I mistakenly named those in convert_new

        return Dataset(array_images, self.image_shape[0], array_embeddings,
                       list_filenames, self.workdir, None,
                       aug_flag, class_id, class_range)
