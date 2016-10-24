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

from infogan.misc.datasets_embedding_new import TextDataset
from infogan.misc.datasets_embedding import AttributeDataset
from infogan.models.v_regularized_gan import ConRegularizedGAN
from infogan.algos.v_infogan_trainer import ConInfoGANTrainer
from infogan.misc.utils import mkdir_p
from infogan.misc.config import cfg, cfg_from_file


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

    datadir = 'Data/%s' % cfg.DATASET_NAME
    filename_test = '%s/%s/test' % (datadir, cfg.FILENAME)
    filename_train = '%s/%s/train' % (datadir, cfg.FILENAME)
    if cfg.ENCODER_INPUT == 'attribute':
        dataset = AttributeDataset(datadir)
        dataset.test, dataset.fixedvisual_test = dataset.get_data(filename_test, 10, 'test')
        if cfg.TRAIN.FLAG:
            dataset.train, dataset.fixedvisual_train = dataset.get_data(filename_train, 20, 'train')
    elif cfg.ENCODER_INPUT == 'text':
        dataset = TextDataset(datadir)
        dataset.test = dataset.get_data(filename_test)
        if cfg.TRAIN.FLAG:
            dataset.train = dataset.get_data(filename_train)
    else:
        NotImplementedError
    # for i in range(dataset.fixedvisual_train.embeddings.shape[0]):
    #     print(dataset.fixedvisual_train.embeddings[i])

    if cfg.TRAIN.FLAG:
        ckt_logs_dir = "ckt_logs/%s_tmp/%s_%s" % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        mkdir_p(ckt_logs_dir)
    else:
        s_tmp = cfg.TRAIN.PRETRAINED_MODEL
        ckt_logs_dir = s_tmp[:s_tmp.find('.ckpt')]
        mkdir_p(ckt_logs_dir)

    model = ConRegularizedGAN(
        image_shape=dataset.image_shape
    )

    algo = ConInfoGANTrainer(
        model=model,
        dataset=dataset,
        ckt_logs_dir=ckt_logs_dir
    )

    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        algo.test()
