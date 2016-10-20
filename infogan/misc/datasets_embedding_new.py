from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import pickle
import random
from sklearn import preprocessing

from infogan.misc.convert_new import get_class_image_list

class Dataset(object):
    def __init__(self, images, hr_images=None, embeddings=None,
                 filenames=None, workdir=None,
                 bg_images=None, labels=None, aug_flag=True,
                 class_id=None, class_range=None):
        self._images = images
        self._hr_images = hr_images
        self._embeddings = embeddings
        self._filenames = filenames
        self.workdir = workdir
        self._bg_images = bg_images
        self._labels = labels
        self._epochs_completed = -1
        self._num_examples = len(images)
        # shuffle on first run
        self._index_in_epoch = self._num_examples
        self._aug_flag = aug_flag
        self._class_id = np.array(class_id)
        self._class_range = class_range
        self._default_size = 64
        self._perm = None

    @property
    def images(self):
        return self._images

    @property
    def hr_images(self):
        return self._hr_images

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
    def bg_images(self):
        return self._bg_images

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def transform(self, images, imsize):
        if self._aug_flag:
            transformed_images = np.zeros([images.shape[0], imsize, imsize, 3])
            ori_size = images.shape[1]
            for i in range(images.shape[0]):
                h1 = int(np.floor((ori_size - imsize) * np.random.random()))
                w1 = int(np.floor((ori_size - imsize) * np.random.random()))
                cropped_image = images[i][w1: w1 + imsize, h1: h1 + imsize, :]
                if random.random() > 0.5:
                    transformed_images[i] = np.fliplr(cropped_image)
                else:
                    transformed_images[i] = cropped_image
            return transformed_images
        else:
            return images

    def sample_embeddings_backup(self, embeddings, filenames, window=2):
        if len(embeddings.shape) == 2 or embeddings.shape[1] == 1:
            return np.squeeze(embeddings)
        else:
            batch_size, embedding_num, _ = embeddings.shape
            # randix = np.random.randint(embedding_num, size=batch_size)
            # return np.squeeze(embeddings[np.arange(batch_size), randix, :])
            # Take every 5 captions to compute the mean vector
            sampled_embeddings = []
            sampled_captions = []
            randix = np.random.randint(window, embedding_num - window, size=batch_size)
            for i in range(batch_size):
                with open(self.workdir + '/text_c10/' + filenames[i] + '.txt', "r") as f:
                    captions = f.read().split('\n')
                captions = [cap for cap in captions if len(cap) > 0]
                # print(captions)
                sampled_captions.append(captions[randix[i]])
                if window > 10:
                    sampled_embeddings.append(embeddings[i, randix[i], :])
                else:
                    # sampled_embeddings.append(np.mean(embeddings[i, (randix[i] - window):(randix[i] + window), :], axis=0))
                    e_sample = embeddings[i, (randix[i] - window):(randix[i] + window), :]
                    e_mean = np.mean(e_sample, axis=0)
                    sampled_embeddings.append(e_mean)
            return np.array(sampled_embeddings), sampled_captions

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

    def sample_bg_images(self, bg_images):
        # print(bg_images.shape)
        if len(bg_images.shape) == 4:
            return np.squeeze(bg_images)
        else:  # len(bg_images.shape) == 5
            batch_size, bg_num, _, _, _ = bg_images.shape
            randix = np.random.randint(bg_num, size=batch_size)
            return np.squeeze(bg_images[np.arange(batch_size), randix, :])

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
        sampled_images = self.transform(sampled_images, self._default_size)
        sampled_wrong_images = self.transform(sampled_wrong_images, self._default_size)
        ret_list = [sampled_images, sampled_wrong_images]

        if self._hr_images is not None:
            sampled_hr_images = self._hr_images[current_ids]
            sampled_hr_images = self.transform(sampled_hr_images, self._default_size * 2)
            ret_list.append(sampled_hr_images)
        else:
            ret_list.append(None)
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
        if self._bg_images is not None:
            sampled_bg_images = self.sample_bg_images(self._bg_images[current_ids])
            ret_list.append(sampled_bg_images)
        else:
            ret_list.append(None)
        if self._labels is not None:
            ret_list.append(self._labels[current_ids])
        else:
            ret_list.append(None)
        return ret_list


class TextDataset(object):
    def __init__(self, workdir):
        self.image_shape = [64, 64, 3]
        self.image_dim = 64 * 64 * 3
        self.embedding_shape = [1024]
        self.train = None
        self.test = None
        self.workdir = workdir

    def get_data(self, pickle_path, aug_flag=True):
        # with open(pickle_path + '/64images.pickle', 'rb') as f:
        if aug_flag:
            with open(pickle_path + '/76images.pickle', 'rb') as f:
                images = pickle.load(f)
                array_images = np.array([image for image in images])
                print('array_images: ', array_images.shape)

        else:
            with open(pickle_path + '/64images.pickle', 'rb') as f:
                images = pickle.load(f)
                array_images = np.array([image for image in images])
                print('array_images: ', array_images.shape)
                _, height, width, depth = array_images.shape
                self.image_dim = height * width * depth
                self.image_shape = [height, width, depth]

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

        array_hr_images = None
        with open(pickle_path + '/152images.pickle', 'rb') as f:
            images = pickle.load(f)
            array_hr_images = np.array([image for image in images])
            print('array_hr_images: ', array_hr_images.shape)

        return Dataset(array_images, array_hr_images, array_embeddings,
                       list_filenames, self.workdir, None, None,
                       aug_flag, class_id, class_range)


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
