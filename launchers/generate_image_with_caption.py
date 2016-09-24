from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np
import argparse
from os.path import join
import scipy.misc
import os
import pickle

from infogan.models.c_regularized_gan import ConRegularizedGAN
from infogan.misc.distributions import Uniform, Categorical, LatentGaussian, MeanBernoulli
from infogan.models.c_regularized_gan import ConRegularizedGAN
from infogan.algos.c_infogan_trainer import ConInfoGANTrainer
from infogan.misc.utils import mkdir_p
from infogan.misc.datasets_embedding import FlowerDataset
from infogan.embeddings import skipthoughts


def read_captions(file_name):
    with open(file_name) as f:
        captions = f.read().split('\n')
    captions = [cap for cap in captions if len(cap) > 0]
    print(captions)
    model = skipthoughts.load_model()
    caption_vectors = skipthoughts.encode(model, captions)
    return captions, caption_vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/home/han/Documents/CVPR2017/InfoGAN/",
                        help='Data Directory')
    parser.add_argument('--model_path', type=str, default="""
/home/han/Documents/CVPR2017/InfoGAN/ckt/flower/
flower_2016_09_23_01_23_24/flower_2016_09_23_01_23_24_30000.ckpt""".replace('\n', ''),
                        help='Trained Model Path')
    parser.add_argument('--n_images', type=int, default=10,
                        help='Number of Images per Caption')
    parser.add_argument('--caption_file', type=str, default='Data/sample_captions.txt',
                        help='caption file')
    parser.add_argument('--embedding_file', type=str, default='Data/sample_embedding.pickle',
                        help='embedding_file')
    args = parser.parse_args()
    if os.path.exists(args.embedding_file):
        with open(args.embedding_file, 'rb') as f:
            captions, embeddings = pickle.load(f)
    else:
        captions, embeddings = read_captions(args.caption_file)
        with open(args.embedding_file, 'wb') as f_out:
            pickle.dump([captions, embeddings], f_out)

    # batch_size = 128
    test_num = embeddings.shape[0]
    embedding_dim = 100

    latent_spec = [
        (Uniform(64), False),
    ]
    con_latent_spec = [
        (LatentGaussian(embedding_dim), True)
    ]

    file_name = '/home/han/Documents/CVPR2017/data/flowers/flowers64test.pickle'
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

    caption_image_dic = {}
    mkdir_p(join(args.data_dir, 'val_samples_text'))

    # for cn, caption_vector in enumerate(caption_vectors):
    for cn in range(test_num):
        array_embeddings = np.tile(embeddings[cn], (args.n_images, 1))
        [gen_image] = sess.run([generated_images],
                               feed_dict={con_embedding: array_embeddings})
        gen_image = (gen_image + 1.0) * 255.0 / 2
        gen_image = gen_image.astype('uint8')
        caption_images = []
        caption_images = [gen_image[i, :, :, :] for i in range(0, args.n_images)]
        caption_image_dic[cn] = caption_images
        print("Generating %d image" % cn)

    for f in os.listdir(join(args.data_dir, 'val_samples_text')):
        if os.path.isfile(f):
            os.unlink(join(args.data_dir, 'val_samples_text/' + f))

    for cn in range(test_num):
        caption_images = []
        print(captions[cn])
        for i, im in enumerate(caption_image_dic[cn]):
            caption_images.append(im)
            caption_images.append(np.zeros((64, 5, 3)))
        combined_image = np.concatenate(caption_images[0:-1], axis=1)
        scipy.misc.imsave(join(args.data_dir, 'val_samples_text/combined_image_{}.jpg'.format(cn)), combined_image)


if __name__ == '__main__':
    main()
