from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import prettytensor as pt
import tensorflow as tf
import numpy as np
import sys
from six.moves import range
from progressbar import ETA, Bar, Percentage, ProgressBar
from infogan.misc.distributions import Bernoulli, Gaussian, Categorical

TINY = 1e-8


def sampleGaussian(mu, log_sigma):
    """(Differentiably!) draw sample from Gaussian with given shape, subject to random noise epsilon"""
    with tf.name_scope("sample_gaussian"):
        # reparameterization trick
        epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
        return mu + epsilon * tf.exp(log_sigma)


# reduce_mean normalize also the dimension of the embeddings
def KL_loss(mu, log_sigma):
    with tf.name_scope("KL_divergence"):
        loss = tf.reduce_mean(-log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu)))
        return loss




class ConInfoGANTrainer(object):
    def __init__(self,
                 model,
                 batch_size,
                 dataset=None,
                 exp_name="experiment",
                 log_dir="logs",
                 checkpoint_dir="ckt",
                 pretrained_model=None,
                 max_epoch=100,
                 updates_per_epoch=50,
                 snapshot_interval=2000,
                 info_reg_coeff=1.0,
                 con_info_reg_coeff=1.0,
                 discriminator_learning_rate=2e-4,
                 generator_learning_rate=2e-4,
                 encoder_learning_rate=2e-4
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.model_path = pretrained_model
        self.snapshot_interval = snapshot_interval
        self.updates_per_epoch = updates_per_epoch
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.encoder_learning_rate = encoder_learning_rate
        self.info_reg_coeff = info_reg_coeff
        self.con_info_reg_coeff = con_info_reg_coeff
        self.discriminator_trainer = None
        self.generator_trainer = None
        self.images = None
        self.embeddings = None
        self.log_vars = []

    def init_opt(self):
        self.images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='real_images')
        self.embeddings = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.embedding_shape,
            name='conditional_embeddings'
        )
        self.z_noise = tf.placeholder(
            tf.float32, [self.batch_size] + [self.model.ef_dim]
        )
        self.z_noise_c_var = tf.placeholder(
            tf.float32, [self.batch_size] + [self.model.ef_dim]

        )

        recon_vs_gan = 1e-6

        with pt.defaults_scope(phase=pt.Phase.train):
            # z_var = self.model.latent_dist.sample_prior(self.batch_size)
            c_var = self.model.generate_condition(self.embeddings)
            # self.log_vars.append(("condition_mean", c_var[0]))
            # self.log_vars.append(("condition_log_sigma", c_var[1]))
            # z_c_var = tf.concat(1, [z_var, c_var])
            # z_c_var = self.model.con_latent_dist.sample_prior_with_condition(
                # self.batch_size, c_var[0], c_var[1])

            kl_loss = KL_loss(c_var[0], c_var[1])
            z_c_var = c_var[0] + c_var[1] * self.z_noise_c_var

            fake_x = self.model.generate(z_c_var)
            noise_x = self.model.generate(self.z_noise)

            real_d, real_f, real_e = self.model.discriminate(self.images)
            fake_d, fake_f, fake_e = self.model.discriminate(fake_x)
            noise_d, _, _ = self.model.discriminate(noise_x)

            like_loss = tf.reduce_mean(tf.square(real_f - fake_f)) / 2.

            d_loss_legit = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_d, tf.ones_like(real_d)))
            d_loss_fake1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_d, tf.zeros_like(fake_d)))
            d_loss_fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(noise_d, tf.zeros_like(noise_d)))
            d_loss_fake = d_loss_fake1 + d_loss_fake2
            d_loss = d_loss_legit + d_loss_fake
            discriminator_loss = d_loss

            g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_d, tf.ones_like(fake_d)))
            g_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(noise_d, tf.ones_like(noise_d)))
            g_loss = g_loss1 + g_loss2 + recon_vs_gan * like_loss
            generator_loss = g_loss

            e_loss = kl_loss + like_loss
            encoder_loss = e_loss


            self.log_vars.append(("discriminator_loss", discriminator_loss))
            self.log_vars.append(("generator_loss", generator_loss))


            all_vars = tf.trainable_variables()

            e_vars = [var for var in all_vars if
                      var.name.startswith('c_')]
            d_vars = [var for var in all_vars if
                      var.name.startswith('d_')]
            g_vars = [var for var in all_vars if
                      var.name.startswith('g_')]

            self.log_vars.append(("max_real_d", tf.reduce_max(real_d)))
            self.log_vars.append(("min_real_d", tf.reduce_min(real_d)))
            self.log_vars.append(("max_fake_d", tf.reduce_max(fake_d)))
            self.log_vars.append(("min_fake_d", tf.reduce_min(fake_d)))

            discriminator_optimizer = tf.train.AdamOptimizer(self.discriminator_learning_rate, beta1=0.5)
            self.discriminator_trainer = pt.apply_optimizer(discriminator_optimizer, losses=[discriminator_loss],
                                                            var_list=d_vars)

            generator_optimizer = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=0.5)
            self.generator_trainer = pt.apply_optimizer(generator_optimizer, losses=[generator_loss], var_list=g_vars)

            encoder_optimizer = tf.train.AdamOptimizer(self.encoder_learning_rate, beta1=0.5)
            self.encoder_trainer = pt.apply_optimizer(encoder_optimizer, losses=[encoder_loss], var_list=e_vars)

            for k, v in self.log_vars:
                if k == 'condition':
                    tf.histogram_summary(k, v)
                else:
                    tf.scalar_summary(k, v)

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("model", reuse=True) as scope:
                # self.visualize_all_factors()
                print("success")

    def visualize_one_factor(self, fixed_noncat, cur_cat, filename):
        if cur_cat is None:
            z_var = fixed_noncat
        else:
            z_var = tf.constant(np.concatenate([fixed_noncat, cur_cat], axis=1))

        if (len(self.model.con_latent_dist.dists) > 0):
            c_var = self.model.generate_condition(self.embeddings)
            z_c_var = tf.concat(1, [z_var, c_var])

            fake_x = self.model.generate(z_c_var)
        else:
            fake_x = self.model.generate(z_var)

        img_var = fake_x

        rows = 10
        img_var = tf.reshape(img_var, [self.batch_size] + list(self.dataset.image_shape))
        img_var = img_var[:rows * rows, :, :, :]
        imgs = tf.reshape(img_var, [rows, rows] + list(self.dataset.image_shape))
        stacked_img = []
        for row in range(rows):
            row_img = []
            row_img.append(self.images[row * rows, :, :, :])  # real image
            for col in range(rows):
                row_img.append(imgs[row, col, :, :, :])
            # each rows is 1realimage +10_fakeimage
            stacked_img.append(tf.concat(1, row_img))
        imgs = tf.concat(0, stacked_img)
        imgs = tf.expand_dims(imgs, 0)
        tf.image_summary(filename, imgs)

    def visualize_all_factors(self):
        fixed_noncat = np.concatenate([
            np.tile(
                self.model.nonreg_latent_dist.sample_prior(10).eval(),
                [10, 1]
            ),
            self.model.nonreg_latent_dist.sample_prior(self.batch_size - 100).eval(),
        ], axis=0)

        fixed_cat = None
        if (len(self.model.reg_latent_dist.dists) > 0):
            fixed_cat = np.concatenate([
                np.tile(
                    self.model.reg_latent_dist.sample_prior(10).eval(),
                    [10, 1]
                ),
                self.model.reg_latent_dist.sample_prior(self.batch_size - 100).eval(),
            ], axis=0)

        # fixed both nonreg_latent_dist and reg_latent_dist for each column,
        # but con_latent_dist is different
        self.visualize_one_factor(fixed_noncat, fixed_cat, "image_fixedall")

        # fixed nonreg_latent_dist but different reg_latent_dist
        # and con_latent_dist for each column
        offset = 0
        for dist_idx, dist in enumerate(self.model.reg_latent_dist.dists):
            cur_cat = None
            if isinstance(dist, Gaussian):
                assert dist.dim == 1, "Only dim=1 is currently supported"
                c_vals = []
                for idx in xrange(10):
                    c_vals.extend([-1.0 + idx * 2.0 / 9] * 10)
                c_vals.extend([0.] * (self.batch_size - 100))
                vary_cat = np.asarray(c_vals, dtype=np.float32).reshape((-1, 1))
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset + 1] = vary_cat
                offset += 1
            elif isinstance(dist, Categorical):
                lookup = np.eye(dist.dim, dtype=np.float32)
                cat_ids = []
                for idx in xrange(10):
                    cat_ids.extend([idx] * 10)
                cat_ids.extend([0] * (self.batch_size - 100))
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset + dist.dim] = lookup[cat_ids]
                offset += dist.dim
            elif isinstance(dist, Bernoulli):
                assert dist.dim == 1, "Only dim=1 is currently supported"
                lookup = np.eye(dist.dim, dtype=np.float32)
                cat_ids = []
                for idx in xrange(10):
                    cat_ids.extend([int(idx / 5)] * 10)
                cat_ids.extend([0] * (self.batch_size - 100))
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset + dist.dim] = np.expand_dims(np.array(cat_ids), axis=-1)
                # import ipdb; ipdb.set_trace()
                offset += dist.dim
            else:
                raise NotImplementedError

            filename = "image_%d_%s" % (dist_idx, dist.__class__.__name__)
            self.visualize_one_factor(fixed_noncat, cur_cat, filename)

    def preprocess(self, embeddings):
        # make sure every row with 10 column have the same embeddings
        for i in range(10):
            for j in range(1, 10):
                embeddings[i * 10 + j] = embeddings[i * 10]
        return embeddings

    def train(self):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            with tf.device("/gpu:0"):
                self.init_opt()
                init = tf.initialize_all_variables()
                sess.run(init)

                summary_op = tf.merge_all_summaries()
                summary_writer = tf.train.SummaryWriter(self.log_dir, sess.graph)

                saver = tf.train.Saver()

                if self.model_path:
                    print("Reading model parameters from %s" % self.model_path)
                    saver.restore(sess, self.model_path)
                    counter = self.model_path[self.model_path.rfind('_') + 1:self.model_path.rfind('.')]
                    counter = int(counter)
                else:
                    print("Created model with fresh parameters.")
                    sess.run(tf.initialize_all_variables())
                    counter = 0

                # log_vars = [x for _, x in self.log_vars]
                # log_keys = [x for x, _ in self.log_vars]
                log_keys = []
                log_vars = []
                for k, v in self.log_vars:
                    if k == 'condition':
                        condition_var = [v]
                    else:
                        log_keys.append(k)
                        log_vars.append(v)

                for epoch in range(int(counter / self.updates_per_epoch), self.max_epoch):
                    widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(maxval=self.updates_per_epoch, widgets=widgets)
                    pbar.start()

                    all_log_vals = []
                    for i in range(self.updates_per_epoch):
                        pbar.update(i)
                        images, embeddings, _ = self.dataset.train.next_batch(self.batch_size)
                        z1 = np.random.normal(0., 1., (self.batch_size, self.model.ef_dim))
                        z2 = np.random.normal(0., 1., (self.batch_size, self.model.ef_dim))
                        feed_dict = {self.images: images,
                                     self.embeddings: embeddings,
                                     self.z_noise_c_var: z1,
                                     self.z_noise: z2}
                        log_vals = sess.run([self.discriminator_trainer] + log_vars, feed_dict)[1:]
                        sess.run(self.generator_trainer, feed_dict)
                        sess.run(self.encoder_trainer, feed_dict)

                        all_log_vals.append(log_vals)
                        counter += 1

                        if counter % self.snapshot_interval == 0:
                            snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                            fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                            print("Model saved in file: %s" % fn)

                    images, embeddings, _ = self.dataset.train.next_batch(self.batch_size)
                    embeddings = self.preprocess(embeddings)
                    z1 = np.random.normal(0., 1., (self.batch_size, self.model.ef_dim))
                    z2 = np.random.normal(0., 1., (self.batch_size, self.model.ef_dim))
                    feed_dict = {self.images: images,
                                 self.embeddings: embeddings,
                                 self.z_noise_c_var: z1,
                                 self.z_noise: z2}
                    summary_str = sess.run(summary_op, feed_dict)
                    summary_writer.add_summary(summary_str, counter)

                    avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                    log_dict = dict(zip(log_keys, avg_log_vals))

                    log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))
                    print("Epoch %d | " % (epoch) + log_line)
                    sys.stdout.flush()
                    if np.any(np.isnan(avg_log_vals)):
                        raise ValueError("NaN detected!")
