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
            tf.float32, [self.batch_size] + [self.model.ef_dim],
            name='sample_noise'
        )
        self.z_noise_c_var = tf.placeholder(
            tf.float32, [self.batch_size] + [self.model.ef_dim],
            name='sample_c_var'
        )

        recon_vs_gan = 1e-6

        with pt.defaults_scope(phase=pt.Phase.train):
            c_var = self.model.generate_condition(self.embeddings)
            self.log_vars.append(("condition_mean", c_var[0]))
            self.log_vars.append(("condition_log_sigma", c_var[0]))

            kl_loss = KL_loss(c_var[0], c_var[1])
            z_c_var = c_var[0] + c_var[1] * self.z_noise_c_var

            fake_x = self.model.get_generator(z_c_var)
            noise_x = self.model.get_generator(self.z_noise)

            # Change by TX (4)
            real_shared_layers = self.model.get_discriminator_shared(self.images)
            fake_shared_layers = self.model.get_discriminator_shared(fake_x)
            noise_shared_layers = self.model.get_discriminator_shared(noise_x)

            real_d = self.model.get_discriminator(real_shared_layers)
            fake_d = self.model.get_discriminator(fake_shared_layers)
            noise_d = self.model.get_discriminator(noise_shared_layers)

            # ##like loss based on feature distance between real and fake images##
            # real_f = self.model.extract_features(real_shared_layers)
            # fake_f = self.model.extract_features(fake_shared_layers)
            # like_loss = tf.reduce_mean(tf.square(real_f - fake_f)) / 2.
            # ##like loss based on distance between text embedding and reconstructed embedding##
            real_t = self.model.reconstuct_context(real_shared_layers)
            fake_t = self.model.reconstuct_context(fake_shared_layers)
            real_like = tf.reduce_mean(tf.square(real_t - self.embeddings)) / 2.
            fake_like = tf.reduce_mean(tf.square(fake_t - self.embeddings)) / 2.
            like_loss = 100. * (real_like + fake_like) / 2.
            # **********

            d_loss_legit = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_d, tf.ones_like(real_d)))
            d_loss_fake1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_d, tf.zeros_like(fake_d)))
            d_loss_fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(noise_d, tf.zeros_like(noise_d)))
            # Change by TX (1)
            # d_loss_fake = d_loss_fake1 + d_loss_fake2
            d_loss_fake = (d_loss_fake1 + d_loss_fake2) / 2.
            # **********
            d_loss = d_loss_legit + d_loss_fake
            discriminator_loss = d_loss

            g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_d, tf.ones_like(fake_d)))
            g_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(noise_d, tf.ones_like(noise_d)))
            # Change by TX (1)
            # g_loss = g_loss1 + g_loss2 + recon_vs_gan * like_loss
            g_loss = (g_loss1 + g_loss2) / 2.
            # **********
            generator_loss = g_loss

            e_loss = kl_loss + like_loss
            encoder_loss = e_loss

            self.log_vars.append(("discriminator_loss", discriminator_loss))
            self.log_vars.append(("generator_loss", generator_loss))
            self.log_vars.append(("encoder_loss", encoder_loss))

            self.log_vars.append(("d_loss_fake", d_loss_fake1))
            self.log_vars.append(("d_loss_noise", d_loss_fake2))
            self.log_vars.append(("d_loss_real", d_loss_legit))

            self.log_vars.append(("e_kl_loss", kl_loss))
            self.log_vars.append(("e_like_loss", like_loss))
            self.log_vars.append(("g_loss_fake", g_loss1))

            self.log_vars.append(("g_loss_noise", g_loss2))
            self.log_vars.append(("g_like_loss_reweight", recon_vs_gan * like_loss))

            all_vars = tf.trainable_variables()

            e_vars = [var for var in all_vars if
                      var.name.startswith('c_')]
            d_vars = [var for var in all_vars if
                      var.name.startswith('d_')]
            g_vars = [var for var in all_vars if
                      var.name.startswith('g_')]

            real_p = tf.nn.sigmoid(real_d)
            feak_p = tf.nn.sigmoid(fake_d)
            self.log_vars.append(("max_real_p", tf.reduce_max(real_p)))
            self.log_vars.append(("min_real_p", tf.reduce_min(real_p)))
            self.log_vars.append(("max_fake_p", tf.reduce_max(feak_p)))
            self.log_vars.append(("min_fake_p", tf.reduce_min(feak_p)))

            discriminator_optimizer = tf.train.AdamOptimizer(self.discriminator_learning_rate, beta1=0.5)
            self.discriminator_trainer = pt.apply_optimizer(discriminator_optimizer, losses=[discriminator_loss],
                                                            var_list=d_vars)

            # Change by TX (2)
            generator_optimizer = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=0.5)
            self.generator_trainer = pt.apply_optimizer(generator_optimizer, losses=[generator_loss + encoder_loss],
                                                        var_list=g_vars + e_vars)
            # self.generator_trainer = pt.apply_optimizer(generator_optimizer, losses=[generator_loss], var_list=g_vars)

            # encoder_optimizer = tf.train.AdamOptimizer(self.encoder_learning_rate, beta1=0.5)
            # self.encoder_trainer = pt.apply_optimizer(encoder_optimizer, losses=[encoder_loss], var_list=e_vars)
            # ****************************

            all_sum = {}
            for k, v in self.log_vars:
                if k.startswith('condition'):
                    all_sum[k] = tf.histogram_summary(k, v)
                else:
                    all_sum[k] = tf.scalar_summary(k, v)
            self.g_sum = tf.merge_summary([all_sum['g_loss_fake'], all_sum['g_loss_noise'],
                                           all_sum['g_like_loss_reweight'], all_sum['generator_loss']])
            self.d_sum = tf.merge_summary([all_sum['d_loss_fake'], all_sum['d_loss_noise'],
                                           all_sum['d_loss_real'], all_sum['discriminator_loss']])
            self.e_sum = tf.merge_summary([all_sum['e_kl_loss'], all_sum['e_like_loss'],
                                           all_sum['encoder_loss']])
            self.other_sum = tf.merge_summary([all_sum['max_real_p'], all_sum['min_real_p'],
                                               all_sum['max_fake_p'], all_sum['min_fake_p'],
                                               all_sum['condition_mean'], all_sum['condition_log_sigma']])

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("model", reuse=True) as scope:
                self.visualization()
                print("success")

    def visualize_one_superimage(self, img_var, images, rows, filename):
        stacked_img = []
        for row in range(rows):
            row_img = []
            row_img.append(images[row * rows, :, :, :])  # real image
            for col in range(rows):
                row_img.append(img_var[row * rows + col, :, :, :])
            # each rows is 1realimage +10_fakeimage
            stacked_img.append(tf.concat(1, row_img))
        imgs = tf.concat(0, stacked_img)
        imgs = tf.expand_dims(imgs, 0)
        current_img_summary = tf.image_summary(filename, imgs)
        return current_img_summary

    def visualization(self):
        c_var = self.model.generate_condition(self.embeddings)
        z_c_var = c_var[0] + c_var[1] * self.z_noise_c_var
        img_c_var = self.model.get_generator(z_c_var)
        img_sum1 = self.visualize_one_superimage(img_c_var[:64, :, :, :], self.images[:64, :, :, :], 8, "train_image_on_text")
        img_sum2 = self.visualize_one_superimage(img_c_var[64:128, :, :, :], self.images[64:128, :, :, :], 8, "test_image_on_text")

        img_noise = self.model.get_generator(self.z_noise)
        img_sum3 = self.visualize_one_superimage(img_noise[:64, :, :, :], self.images[:64, :, :, :], 8, "image_on_noise")
        self.image_summary = tf.merge_summary([img_sum1, img_sum2, img_sum3])

    def preprocess(self, embeddings):
        # make sure every row with 10 column have the same embeddings
        for i in range(8):
            for j in range(1, 8):
                embeddings[i * 8 + j] = embeddings[i * 8]
                # images[i * 8 + j] = images[i * 8]
        return embeddings

    def train(self):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            with tf.device("/gpu:0"):
                self.init_opt()
                init = tf.initialize_all_variables()
                sess.run(init)

                # summary_op = tf.merge_all_summaries()
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

                log_keys = []
                log_vars = []
                condition_var = []
                for k, v in self.log_vars:
                    if k.startswith('condition'):
                        condition_var.append(v)
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
                        # training d
                        images, embeddings, _ = self.dataset.train.next_batch(self.batch_size)
                        z1 = np.random.normal(0., 1., (self.batch_size, self.model.ef_dim))
                        z2 = np.random.normal(0., 1., (self.batch_size, self.model.ef_dim))
                        feed_dict_d = {self.images: images,
                                       self.embeddings: embeddings,
                                       self.z_noise_c_var: z1,
                                       self.z_noise: z2}
                        _, d_summary, log_vals, condition_vals, other_summary = sess.run(
                            [self.discriminator_trainer, self.d_sum,
                             log_vars, condition_var, self.other_sum], feed_dict_d)

                        summary_writer.add_summary(d_summary, counter)
                        summary_writer.add_summary(other_summary, counter)
                        # training g&e
                        images, embeddings, _ = self.dataset.train.next_batch(self.batch_size)
                        z1 = np.random.normal(0., 1., (self.batch_size, self.model.ef_dim))
                        z2 = np.random.normal(0., 1., (self.batch_size, self.model.ef_dim))
                        feed_dict_ge = {self.images: images,
                                        self.embeddings: embeddings,
                                        self.z_noise_c_var: z1,
                                        self.z_noise: z2}
                        # Change by TX (2)
                        _, g_summary, e_summary = sess.run(
                            [self.generator_trainer, self.g_sum, self.e_sum], feed_dict_ge
                        )
                        # _, g_summary = sess.run(
                        #     [self.generator_trainer, self.g_sum], feed_dict_ge
                        # )
                        # _, e_summary = sess.run(
                        #     [self.encoder_trainer, self.e_sum], feed_dict_ge
                        # )
                        # *****************
                        summary_writer.add_summary(g_summary, counter)
                        summary_writer.add_summary(e_summary, counter)

                        all_log_vals.append(log_vals)
                        counter += 1

                        if counter % self.snapshot_interval == 0:
                            snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                            fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                            print("Model saved in file: %s" % fn)

                    images_train, embeddings_train, _ = self.dataset.train.next_batch(64)
                    embeddings_train = self.preprocess(embeddings_train)

                    images_test, embeddings_test, _ = self.dataset.test.next_batch(64)
                    embeddings_test = self.preprocess(embeddings_test)

                    images = np.concatenate([images_train, images_test], axis=0)
                    embeddings = np.concatenate([embeddings_train, embeddings_test], axis=0)

                    z1 = np.tile(np.random.normal(0., 1., (8, self.model.ef_dim)), [16, 1])
                    z2 = np.random.normal(0., 1., (self.batch_size, self.model.ef_dim))
                    if self.batch_size > 128:
                        z_pad = np.random.normal(0., 1., (self.batch_size - 128, self.model.ef_dim))
                        z1 = np.concatenate([z1, z_pad], axis=0)
                        images_pad, embeddings_pad, _ = self.dataset.test.next_batch(self.batch_size - 128)
                        images = np.concatenate([images, images_pad], axis=0)
                        embeddings = np.concatenate([embeddings, embeddings_pad], axis=0)

                    feed_dict = {self.images: images,
                                 self.embeddings: embeddings,
                                 self.z_noise_c_var: z1,
                                 self.z_noise: z2}
                    epoch_image_summary = sess.run(self.image_summary, feed_dict)
                    summary_writer.add_summary(epoch_image_summary, counter)

                    avg_log_vals = np.mean(np.array(all_log_vals), axis=0)

                    log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))
                    print("Epoch %d | " % (epoch) + log_line)
                    sys.stdout.flush()
                    if np.any(np.isnan(avg_log_vals)):
                        raise ValueError("NaN detected!")
