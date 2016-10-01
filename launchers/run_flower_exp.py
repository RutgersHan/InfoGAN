from __future__ import print_function
from __future__ import absolute_import
from infogan.misc.distributions import Uniform, Categorical, LatentGaussian, MeanBernoulli, Bernoulli

import tensorflow as tf
import os
from infogan.misc.datasets_embedding import FlowerDataset, BirdAttribuetDataset
from infogan.models.v_regularized_gan import ConRegularizedGAN
from infogan.algos.v_infogan_trainer import ConInfoGANTrainer
from infogan.misc.utils import mkdir_p
import dateutil
import dateutil.tz
import datetime

if __name__ == "__main__":
    # dataset_name = 'flowers'
    dataset_name = 'birds'

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    # root_log_dir = "logs/" + dataset_name
    # root_checkpoint_dir = "ckt/" + dataset_name
    root_log_dir = "ckt_logs/" + dataset_name
    # pretrained_model = "%s/birds_2016_09_29_10_22_42/birds_2016_09_29_10_22_42_10000.ckpt" % root_log_dir
    pretrained_model = None  # "%s/5like_0.001gl_birds_2016_09_30_16_48_16/birds_2016_09_30_16_48_16_14000.ckpt" % root_log_dir

    batch_size = 256
    updates_per_epoch = 50
    max_epoch = 2000
    embedding_dim = 100
    background_dim = 1

    exp_name = "1like_0.0002gl_subftr_attr%s_%s" % (dataset_name, timestamp)

    log_dir = os.path.join(root_log_dir, exp_name)
    checkpoint_dir = log_dir
    # checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

    mkdir_p(log_dir)
    # mkdir_p(checkpoint_dir)
    datadir = 'Data/%s' % dataset_name
    #
    # dataset = FlowerDataset(datadir)
    # dataset.train = dataset.get_data('%s/%s64image_mask_train.pickle' % (datadir, dataset_name))
    # dataset.test = dataset.get_data('%s/%s64image_mask_test.pickle' % (datadir, dataset_name))
    #
    dataset = BirdAttribuetDataset()
    dataset.train, dataset.fixedvisual_train = dataset.get_data('%s/%s64image_mask_attr_train.pickle'
                                                                % (datadir, dataset_name), 20, 'train')
    dataset.test, dataset.fixedvisual_test = dataset.get_data('%s/%s64image_mask_attr_test.pickle'
                                                              % (datadir, dataset_name), 10, 'test')

    latent_spec = [
        # (Bernoulli(embedding_dim), True)
        # (Uniform(1), False),
        (Categorical(32), True),
    ]

    con_latent_spec = [
        # (LatentGaussian(embedding_dim), True)
        Bernoulli(embedding_dim)
    ]

    model = ConRegularizedGAN(
        output_dist=MeanBernoulli(dataset.image_dim),
        latent_spec=latent_spec,
        con_latent_spec=con_latent_spec,
        batch_size=batch_size,
        image_shape=dataset.image_shape,
        text_dim=dataset.embedding_shape[0],
        network_type="flower",
        ef_dim=embedding_dim,
        bg_dim=background_dim
    )

    algo = ConInfoGANTrainer(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        exp_name=exp_name,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        pretrained_model=pretrained_model,
        max_epoch=max_epoch,
        updates_per_epoch=updates_per_epoch,
        info_reg_coeff=1.0,
        con_info_reg_coeff=1.0
    )

    algo.train()
