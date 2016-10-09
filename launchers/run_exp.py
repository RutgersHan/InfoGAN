from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import os
import dateutil
import dateutil.tz
import datetime
import argparse
import pprint
import sys
import numpy as np

from infogan.misc.distributions import Uniform, Gaussian, Categorical, MeanBernoulli, Bernoulli
from infogan.misc.datasets_embedding import TextDataset, AttributeDataset
from infogan.models.v_regularized_gan import ConRegularizedGAN
from infogan.algos.v_infogan_trainer import ConInfoGANTrainer
from infogan.misc.utils import mkdir_p
from infogan.misc.config import cfg, cfg_from_file


def get_latent_spec():
    latent_spec = []
    if cfg.GAN.LATENT_SPEC.UNIFORM_FLAG:
        latent_spec.append((Uniform(cfg.GAN.LATENT_SPEC.UNIFORM.DIM),
                            cfg.GAN.LATENT_SPEC.UNIFORM.REG))
    if cfg.GAN.LATENT_SPEC.GAUSSIAN_FLAG:
        latent_spec.append((Gaussian(cfg.GAN.LATENT_SPEC.GAUSSIAN.DIM),
                            cfg.GAN.LATENT_SPEC.GAUSSIAN.REG))
    if cfg.GAN.LATENT_SPEC.CAT_FLAG:
        latent_spec.append((Categorical(cfg.GAN.LATENT_SPEC.CAT.DIM),
                            cfg.GAN.LATENT_SPEC.CAT.REG))
    if cfg.GAN.LATENT_SPEC.BERNOULLI_FLAG:
        latent_spec.append((Bernoulli(cfg.GAN.LATENT_SPEC.BERNOULLI.DIM),
                            cfg.GAN.LATENT_SPEC.BERNOULLI.REG))

    con_latent_spec = []
    if cfg.GAN.CON_LATENT_SPEC.UNIFORM_FLAG:
        con_latent_spec.append(Uniform(cfg.GAN.CON_LATENT_SPEC.UNIFORM.DIM))
    if cfg.GAN.CON_LATENT_SPEC.GAUSSIAN_FLAG:
        con_latent_spec.append(Gaussian(cfg.GAN.CON_LATENT_SPEC.GAUSSIAN.DIM))
    if cfg.GAN.CON_LATENT_SPEC.CAT_FLAG:
        con_latent_spec.append(Categorical(cfg.GAN.CON_LATENT_SPEC.CAT.DIM))
    if cfg.GAN.CON_LATENT_SPEC.BERNOULLI_FLAG:
        con_latent_spec.append(Bernoulli(cfg.GAN.CON_LATENT_SPEC.BERNOULLI.DIM))

    return latent_spec, con_latent_spec


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=-1, type=int)
    # if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    print('Using config:')
    pprint.pprint(cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    ckt_logs_dir = "ckt_logs_fg/%s/%s_%s" % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    datadir = 'Data/%s' % cfg.DATASET_NAME
    filename_test = '%s/%s/test' % (datadir, cfg.FILENAME)
    filename_train = '%s/%s/train' % (datadir, cfg.FILENAME)
    if cfg.ENCODER_INPUT == 'attribute':
        dataset = AttributeDataset(datadir)
        dataset.test, dataset.fixedvisual_test = dataset.get_data(filename_test, 10, 'test')
        dataset.train, dataset.fixedvisual_train = dataset.get_data(filename_train, 20, 'train')
    elif cfg.ENCODER_INPUT == 'text':
        dataset = TextDataset(datadir)
        dataset.test = dataset.get_data(filename_test)
        dataset.train = dataset.get_data(filename_train)
    else:
        NotImplementedError
    # for i in range(dataset.fixedvisual_train.embeddings.shape[0]):
    #     print(dataset.fixedvisual_train.embeddings[i])

    mkdir_p(ckt_logs_dir)

    latent_spec, con_latent_spec = get_latent_spec()

    model = ConRegularizedGAN(
        output_dist=MeanBernoulli(dataset.image_dim),
        latent_spec=latent_spec,
        con_latent_spec=con_latent_spec,
        image_shape=dataset.image_shape
    )

    algo = ConInfoGANTrainer(
        model=model,
        dataset=dataset,
        ckt_logs_dir=ckt_logs_dir
    )

    algo.train()
