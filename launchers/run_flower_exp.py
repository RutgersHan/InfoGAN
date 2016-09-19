from __future__ import print_function
from __future__ import absolute_import
from infogan.misc.distributions import Uniform, Categorical, LatentGaussian, MeanBernoulli

import tensorflow as tf
import os
from infogan.misc.datasets_embedding import FlowerDataset
from infogan.models.c_regularized_gan import ConRegularizedGAN
from infogan.algos.c_infogan_trainer import ConInfoGANTrainer
from infogan.misc.utils import mkdir_p
import dateutil
import dateutil.tz
import datetime

if __name__ == "__main__":

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    root_log_dir = "logs/flower"
    root_checkpoint_dir = "ckt/flower"
    batch_size = 128
    updates_per_epoch = 50
    max_epoch = 500
    embedding_dim = 100

    exp_name = "flower_%s" % timestamp

    log_dir = os.path.join(root_log_dir, exp_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

    mkdir_p(log_dir)
    mkdir_p(checkpoint_dir)

    file_name = '/home/han/Documents/CVPR2017/data/flowers/flowers64.pickle'

    dataset = FlowerDataset(file_name)
    dataset.get_data()

    latent_spec = [
        (Uniform(64), False),
        (Categorical(32), True),
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
        max_epoch=max_epoch,
        updates_per_epoch=updates_per_epoch,
        info_reg_coeff=1.0,
        con_info_reg_coeff=1.0,
        generator_learning_rate=1e-3,
        discriminator_learning_rate=2e-4,
    )

    algo.train()
