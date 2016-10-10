from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import tensorflow as tf
import numpy as np
import h5py
import os
import pickle
from utils import get_image, colorize
import scipy.misc
import scipy.io as sio
import pandas as pd

# from glob import glob

# TODO: 1. current label is temporary, need to change according to real label
#       2. Current, only split the data into train, need to handel train, test

IMSIZE = 128
FLOWER_DIR = '/home/han/Documents/CVPR2017/data/flowers'
BIRD_DIR = '/home/tao/deep_learning/CVPR2017/Han/InfoGAN/Data/birds'

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def convert_flowers_dataset_pickle(data_dir, train_ratio=0.75):
    h = h5py.File(os.path.join(data_dir, 'flower_tv.hdf5'))
    outfile = os.path.join(data_dir, 'flowers' + str(IMSIZE) + '.pickle')
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
    outfile = os.path.join(data_dir, 'flowers' + str(IMSIZE) + '.tfrecords')
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

        # [-1, 1]
        image = get_image(f_name, IMSIZE, is_crop=False, resize_w=IMSIZE)
        image = colorize(image)
        assert image.shape == (IMSIZE, IMSIZE, 3)
        # ##Convert image to [0, 255]
        # image += 1.
        # image *= (255. / 2.)
        # image = image.astype('uint8')

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
            'label': _int64_feature(label)}))
        writer.write(example.SerializeToString())
    writer.close()


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


def convert_birds_dataset_pickle(data_dir, train_ratio=0.75):
    h = h5py.File(os.path.join(data_dir, 'bird_tv.hdf5'))

    class_list = []
    image_lists = {}
    captions = {}
    for ds in h.iteritems():
        class_name = ds[0]
        class_list.append(class_name)
        image_lists[class_name] = []
        for ds2 in ds[1].iteritems():
            filename = ds2[0]
            image_path = os.path.join(class_name, filename)
            captions[image_path] = np.array(ds2[1])
            # print(captions[image_path][:10])
            image_lists[class_name].append(image_path)

    class_list.sort()

    training_num = int(len(class_list) * train_ratio)
    training_class_list = class_list[0:training_num]
    test_class_list = class_list[training_num:]
    print(len(training_class_list), len(test_class_list))

    btrain = 1

    height = IMSIZE
    width = IMSIZE
    depth = 3
    embedding_num = 10
    images = []
    masks = []
    embeddings = []
    labels = []

    if btrain:
        class_list = training_class_list
        # segmented image_mask
        outfile = os.path.join(data_dir, 'birds' + str(IMSIZE) + 'image_mask' + '_train.pickle')
    else:
        class_list = test_class_list
        outfile = os.path.join(data_dir, 'birds' + str(IMSIZE) + 'image_mask' + '_test.pickle')
    filename_mask_dic = load_mask(data_dir)

    for i, c in enumerate(class_list):
        for j, f in enumerate(image_lists[c]):
            # ####mask only has 0/1 values
            mask = filename_mask_dic[f]
            mask = scipy.misc.imresize(mask, [IMSIZE, IMSIZE])
            # print(f, mask, np.sum(mask))
            # print(type(mask), mask.shape)
            # return
            masks.append(mask)
            f_name = os.path.join(data_dir, 'images_cropped', f)

            # f_name = os.path.join(data_dir, 'segmented_images_cropped', f)

            # print(f_name)
            # ####[-1, 1]
            image = get_image(f_name, IMSIZE, is_crop=False, resize_w=IMSIZE)
            image = colorize(image)
            # print(image)
            assert image.shape == (IMSIZE, IMSIZE, 3)
            # ######Convert image to [0, 255]
            # image += 1.
            # image *= (255. / 2.)
            # image = image.astype('uint8')

            embedding = captions[f]
            # print(f)
            # print(embedding)

            label = i  # temporary
            print('%d\t%d' % (j, label))
            images.append(image)
            embeddings.append(embedding)
            labels.append(label)

    with open(outfile, 'wb') as f_out:
        pickle.dump([height, width, depth, embedding_num, images, masks,
                     embeddings, labels], f_out)


def crop_background_images(f_name, imsize, img_id):
    img = scipy.misc.imread(f_name)
    img = np.array(img) / 127.5 - 1.
    img = colorize(img)

    # crop background images from the original images
    bg_imgs = []
    h = img.shape[0]
    w = img.shape[1]
    bg_imgs.append(img[:imsize, :imsize, :])  # LT
    bg_imgs.append(img[:imsize, w - imsize:, :])  # RT
    bg_imgs.append(img[h - imsize:, :imsize, :])  # LB
    bg_imgs.append(img[h - imsize:, w - imsize:, :])  # RB
    # scipy.misc.imsave('bg_imgs_%d.jpg' % img_id, np.concatenate(bg_imgs, axis=1))
    # From a list of 3D images to a 4D array [bg_num, h, w, c]
    bg_imgs = np.array(bg_imgs)
    # print(f_name, bg_imgs.shape)
    return bg_imgs


def get_text_embedding(data_dir):
    h = h5py.File(os.path.join(data_dir, 'bird_tv.hdf5'))
    text_captions = {}
    for ds in h.iteritems():
        class_name = ds[0]
        for ds2 in ds[1].iteritems():
            filename = ds2[0]
            image_path = os.path.join(class_name, filename)
            text_captions[image_path] = np.array(ds2[1])
            # print(image_path, text_captions[image_path].shape)
            # print(text_captions[image_path][:10])
    print('text_embedding: ', len(text_captions))
    return text_captions


def get_attribute_vector(data_dir, img_filenames):
    captions = {}
    attributes_strings = {}
    arr_attributes = sio.loadmat(os.path.join(data_dir, 'attributesVec.mat'))['attributesVec']
    print('arr_attributes: ', arr_attributes.shape)
    attributesShortNames = pd.read_csv(os.path.join(data_dir, 'attributesShortNames.txt'),
                                       delim_whitespace=True, header=None)
    for i in xrange(len(img_filenames)):
        name = img_filenames[i]
        captions[name] = arr_attributes[i]
        attributes_strings[name] = []
        s = ''
        idx = np.where(arr_attributes[i, :] == 1)[0]
        # print(type(idx), len(idx), idx)
        if len(idx) > 0:
            df_hasAttr = attributesShortNames.iloc[idx].sort(columns=[1])
            # print(type(df_hasAttr.index), len(df_hasAttr.index), df_hasAttr.index)
            part = ''
            for j in df_hasAttr.index:
                row = attributesShortNames.iloc[j]
                if part != row[1]:
                    # print s
                    part = row[1]
                    s += '\n' + part + ': ' + row[2]
                else:
                    s += ';' + row[2]
            # print s
            attributes_strings[name].append(s)
    return captions, attributes_strings


def get_icml16_text_embedding(data_dir):
    with open(os.path.join(data_dir, 'icml16_text_embedding.p'), 'rb') as f:
        embeddings = pickle.load(f)
        print('text_embedding: ', len(embeddings))
    return embeddings


def get_class_image_list(data_dir, train_ratio):
    # <class_id> <class_name>
    class_list = pd.read_csv(os.path.join(data_dir, 'classes.txt'),
                             delim_whitespace=True, header=None)[1].tolist()
    print('class_list', len(class_list), class_list[:5])
    # Devide into train and test dataset
    class_list.sort()
    training_num = int(len(class_list) * train_ratio)
    training_class_list = class_list[0:training_num]
    test_class_list = class_list[training_num:]
    print('training_classes:', len(training_class_list), 'test_classes', len(test_class_list))

    # <image_id> <image_name>
    img_filenames = pd.read_csv(os.path.join(data_dir, 'images.txt'),
                                delim_whitespace=True, header=None)[1].tolist()
    image_lists = {}
    for class_name in class_list:
        image_lists[class_name] = []
    for i in xrange(len(img_filenames)):
        name = img_filenames[i]
        image_lists[name[:name.find('/')]].append(name)
    return training_class_list, test_class_list, image_lists, img_filenames


def save_data(class_list, image_lists, data_dir, bbox, bimg, bbgimg,
              filename_mask_dic, attr_captions, attr_strings,
              text_captions, icml16_text_captions, blabel, outpath):
    images = []
    masks = []
    bg_images = []
    text_embeddings = []
    icml16_text_embeddings = []
    attr_embeddings = []
    filenames = []
    attributes = []
    labels = []
    for i, c in enumerate(class_list):
        for j, f in enumerate(image_lists[c]):
            # print(f)
            filenames.append(f.replace('.jpg', ''))

            if bimg:
                if bbox:
                    f_name = os.path.join(data_dir, 'images_cropped', f)
                else:
                    f_name = os.path.join(data_dir, 'images', f)
                # ####images [-1, 1] values####################################
                img = get_image(f_name, IMSIZE, is_crop=False, resize_w=IMSIZE)
                images.append(colorize(img))
                # print(images[-1].shape)
                assert images[-1].shape == (IMSIZE, IMSIZE, 3)

            # ####background images [-1, 1] values#########################
            if bbgimg:
                f_name = os.path.join(data_dir, 'images', f)
                # A 4D array [bg_num, h, w, c]
                bg_images.append(crop_background_images(f_name, IMSIZE, j))

            # ####mask only has 0/1 values#####################################
            if filename_mask_dic is not None:
                mask = filename_mask_dic[f]
                masks.append(scipy.misc.imresize(mask, [IMSIZE, IMSIZE]))

            # ####text embeddings (skip-thought vectors) #######################
            if text_captions is not None:
                text_embeddings.append(text_captions[f])
                # print(text_embeddings[-1].shape)
            if icml16_text_captions is not None:
                icml16_text_embeddings.append(icml16_text_captions[f])

            # ####attributes embeddings #######################################
            if attr_captions is not None:
                attr_embeddings.append(attr_captions[f])
                # print(attr_embeddings[-1])
                attributes.append(attr_strings[f])
                # print(attr[-1])

            # ####class labels##################################################
            if blabel:
                labels.append(i)  # temporary
            print('%d\t%d' % (j, i))

    if len(images):
        print('images', len(images), images[0].shape)
        if bbox:
            outfile = outpath + str(IMSIZE) + 'cropped_images.pickle'
        else:
            outfile = outpath + str(IMSIZE) + 'images.pickle'
        with open(outfile, 'wb') as f_out:
            pickle.dump(images, f_out)
            print('save to: ', outfile)
    if len(masks):
        print('masks', len(masks), masks[0].shape)
        if bbox:
            outfile = outpath + str(IMSIZE) + 'cropped_masks.pickle'
        else:
            outfile = outpath + str(IMSIZE) + 'masks.pickle'
        with open(outfile, 'wb') as f_out:
            pickle.dump(masks, f_out)
            print('save to: ', outfile)
    if len(bg_images):
        print('bg_images', len(bg_images), bg_images[0].shape)
        outfile = outpath + str(IMSIZE) + 'bg_images.pickle'
        with open(outfile, 'wb') as f_out:
            pickle.dump(bg_images, f_out)
            print('save to: ', outfile)
    if len(attr_embeddings):
        print('attr_embeddings', len(attr_embeddings), attr_embeddings[0].shape)
        print('attributes', len(attributes), attributes[0])
        outfile = outpath + 'attr_embeddings_names.pickle'
        with open(outfile, 'wb') as f_out:
            pickle.dump([attr_embeddings, attributes], f_out)
            print('save to: ', outfile)
    if len(text_embeddings):
        print('text_embeddings', len(text_embeddings), text_embeddings[0].shape)
        outfile = outpath + 'text_embeddings.pickle'
        with open(outfile, 'wb') as f_out:
            pickle.dump(text_embeddings, f_out)
            print('save to: ', outfile)
    if len(icml16_text_embeddings):
        print('icml16_text_embeddings', len(icml16_text_embeddings),
              icml16_text_embeddings[0].shape)
        outfile = outpath + 'icml16_text_embeddings.pickle'
        with open(outfile, 'wb') as f_out:
            pickle.dump(icml16_text_embeddings, f_out)
            print('save to: ', outfile)
    if len(labels):
        print('labels', len(labels))
        outfile = outpath + 'labels.pickle'
        with open(outfile, 'wb') as f_out:
            pickle.dump(labels, f_out)
            print('save to: ', outfile)
    if len(filenames):
        print('filenames', len(filenames), filenames[0])
        outfile = outpath + 'filenames.pickle'
        with open(outfile, 'wb') as f_out:
            pickle.dump(filenames, f_out)
            print('save to: ', outfile)



def convert_birds_dataset_attribute_pickle(data_dir, train_ratio=0.75):
    bbox = 0

    bimg = 0
    bbgimg = 0
    bmask = 0
    btext = 0
    icml16_btext = 0
    battr = 0
    blabel = 0

    training_class_list, test_class_list,\
        image_lists, img_filenames = get_class_image_list(data_dir, train_ratio)

    # text embeddings
    if btext:
        text_captions = get_text_embedding(data_dir)
    else:
        text_captions = None
    if icml16_btext:
        icml16_text_captions = get_icml16_text_embedding(data_dir)
    else:
        icml16_text_captions = None

    if battr:
        attr_captions, attr_strings = get_attribute_vector(data_dir, img_filenames)
    else:
        attr_captions, attr_strings = [None, None]

    if bmask:
        filename_mask_dic = load_mask(data_dir, bbox)
        print('mask: ', len(filename_mask_dic))
    else:
        filename_mask_dic = None

    # ## For Training data
    save_data(training_class_list, image_lists, data_dir, bbox, bimg, bbgimg,
              filename_mask_dic, attr_captions, attr_strings,
              text_captions, icml16_text_captions,
              blabel, os.path.join(data_dir, 'pickle/train/'))
    # ## For Test data
    save_data(test_class_list, image_lists, data_dir, bbox, bimg, bbgimg,
              filename_mask_dic, attr_captions, attr_strings,
              text_captions, icml16_text_captions,
              blabel, os.path.join(data_dir, 'pickle/test/'))


if __name__ == '__main__':
    # convert_flowers_dataset_pickle(FLOWER_DIR)
    # convert_flowers_dataset_tf(FLOWER_DIR)
    # convert_flowers_test_dataset_pickle(FLOWER_DIR)
    # convert_birds_dataset_pickle(BIRD_DIR)
    convert_birds_dataset_attribute_pickle(BIRD_DIR)
