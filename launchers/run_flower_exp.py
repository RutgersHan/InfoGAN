from __future__ import print_function
from __future__ import absolute_import
from infogan.misc.distributions import Uniform, Categorical, LatentGaussian, MeanBernoulli

import tensorflow as tf
import os
from infogan.misc.datasets_embedding import FlowerDataset
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

    root_log_dir = "logs/" + dataset_name
    root_checkpoint_dir = "ckt/" + dataset_name
    pretrained_model = None  # "ckt/%s/flower_2016_09_25_20_22_49/flower_2016_09_25_20_22_49_12000.ckpt" % dataset_name

    batch_size = 256
    updates_per_epoch = 50
    max_epoch = 2000
    embedding_dim = 100

    exp_name = "%s_%s" % (dataset_name, timestamp)

    log_dir = os.path.join(root_log_dir, exp_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

    mkdir_p(log_dir)
    mkdir_p(checkpoint_dir)

    dataset = FlowerDataset()
    dataset.train = dataset.get_data('Data/%s/%s64train.pickle' % (dataset_name, dataset_name))
    dataset.test = dataset.get_data('Data/%s/%s64test.pickle' % (dataset_name, dataset_name))

    latent_spec = [
        (Uniform(64), False),
        # (Categorical(32), True),
    ]

    con_latent_spec = [
        (LatentGaussian(embedding_dim), True)
    ]

    model = ConRegularizedGAN(
        output_dist=MeanBernoulli(dataset.image_dim),
        latent_spec=latent_spec,
        con_latent_spec=con_latent_spec,
        batch_size=batch_size,
        image_shape=dataset.image_shape,
        network_type="flower",
        ef_dim=embedding_dim
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
