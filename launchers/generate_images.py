from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np
from infogan.models.c_regularized_gan import ConRegularizedGAN
from infogan.misc.distributions import Uniform, Categorical, LatentGaussian, MeanBernoulli
from infogan.models.c_regularized_gan import ConRegularizedGAN
from infogan.algos.c_infogan_trainer import ConInfoGANTrainer
from infogan.misc.utils import mkdir_p
from infogan.misc.datasets_embedding import FlowerDataset

import argparse
import pickle
from os.path import join
import h5py
from Utils import image_processing
import scipy.misc
import random
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="Data",
                        help='Data Directory')
    parser.add_argument('--model_path', type=str, default='Data/Models/latest_model_flowers_temp.ckpt',
                        help='Trained Model Path')
    parser.add_argument('--n_images', type=int, default=5,
                        help='Number of Images per Caption')
    parser.add_argument('--caption_thought_vectors', type=str, default='Data/sample_caption_vectors.hdf5',
                        help='Caption Thought Vector File')
    args = parser.parse_args()

    batch_size = 128
    embedding_dim = 100

    latent_spec = [
        (Uniform(64), False),
        (Categorical(32), True),
    ]
    con_latent_spec = [
        (LatentGaussian(embedding_dim), True)
    ]

    file_name = '/home/han/Documents/CVPR2017/data/flowers/flowers64.pickle'
    dataset = FlowerDataset(file_name)
    dataset.get_data()

    model = ConRegularizedGAN(
        output_dist=MeanBernoulli(dataset.image_dim),
        latent_spec=latent_spec,
        con_latent_spec=con_latent_spec,
        batch_size=args.n_images,
        image_shape=dataset.image_shape,
        network_type="flower",
        ef_dim=embedding_dim
    )
    con_embedding, generated_images = model.generate_for_visualization(
        args.n_images, [4800]
    )
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, args.model_path)

    h = h5py.File(args.caption_thought_vectors)
    caption_vectors = np.array(h['vectors'])
    caption_image_dic = {}

    for cn, caption_vector in enumerate(caption_vectors):
        embeddings = [caption_vector] * args.n_images  # TODO: need to test
        array_embeddings = np.array([
            embedding for embedding in embeddings])
        [gen_image] = sess.run([generated_images],
                               feed_dict={con_embedding: array_embeddings})
        caption_images = []
        caption_images = [gen_image[i, :, :, :] for i in range(0, args.n_images)]
        caption_image_dic[cn] = caption_images
        print("Generating %d image" % cn)

    for f in os.listdir(join(args.data_dir, 'val_samples')):
        if os.path.isfile(f):
            os.unlink(join(args.data_dir, 'val_samples/' + f))

    for cn in range(0, len(caption_vectors)):
        caption_images = []
        for i, im in enumerate(caption_image_dic[cn]):
            caption_images.append(im)
            caption_images.append(np.zeros((64, 5, 3)))
        combined_image = np.concatenate(caption_images[0:-1], axis=1)
    scipy.misc.imsave(join(args.data_dir, 'val_samples/combined_image_{}.jpg'.format(cn)), combined_image)


if __name__ == '__main__':
    main()
