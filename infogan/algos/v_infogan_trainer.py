from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import prettytensor as pt
import tensorflow as tf
import numpy as np
import scipy.misc
import sys
from six.moves import range
from progressbar import ETA, Bar, Percentage, ProgressBar

from infogan.misc.distributions import Bernoulli, Gaussian, Categorical
from infogan.misc.config import cfg

TINY = 1e-8
TYPE_KL_LOSS = 0  # 0: no kl_loss; 1: single kl_losss; 2: global kl_loss
B_MI_LOSS = 0

B_PRETRAIN = 0

B_MASKED = 1


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


def cosine_loss(vector1, vector2):
    normed_v1 = tf.nn.l2_norm(vector1, 1)
    normed_v2 = tf.nn.l2_norm(vector2, 1)
    return tf.contrib.losses.cosine_distance(normed_v1, normed_v2)


def compute_mean_covariance(img):
    shape = img.get_shape()
    batch_size = shape[0].value
    height = shape[1].value
    width = shape[2].value
    channel_num = shape[3].value
    mu = tf.reduce_mean(img, [1, 2], True)

    num_pixels = height * width
    # shape is batch_size * num_pixels * channel_num
    img_hat = img - mu
    img_hat = tf.reshape(img_hat, [batch_size, -1, channel_num])

    # shape is batch_size * channel_num * num_pixels
    img_hat_transpose = tf.transpose(img_hat, perm=[0, 2, 1])
    # shape is batch_size * channle_num * channle_num
    covariance = tf.batch_matmul(img_hat_transpose, img_hat)
    covariance = covariance / tf.to_float(num_pixels)
    return mu, covariance


class ConInfoGANTrainer(object):
    def __init__(self,
                 model,
                 dataset=None,
                 exp_name="model",
                 ckt_logs_dir="ckt_logs",
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.dataset = dataset
        self.exp_name = exp_name
        self.log_dir = ckt_logs_dir
        self.checkpoint_dir = ckt_logs_dir

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.model_path = cfg.TRAIN.PRETRAINED_MODEL

        # self.generator_learning_rate = cfg.TRAIN.GENERATOR_LR
        # self.discriminator_learning_rate = cfg.TRAIN.DISCRIMINATOR_LR

        self.images = None
        self.masks = None
        self.embeddings = None
        self.fg_images = None
        self.log_vars = []

    def build_placeholder(self):
        '''Helper function for init_opt'''
        self.images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='real_images')
        self.wrong_images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='wrong_images'
        )
        self.masks = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape[:2],
            name='real_masks')
        self.embeddings = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.embedding_shape,
            name='conditional_embeddings'
        )

        self.generator_learning_rate = tf.placeholder(
            tf.float32, [],
            name='generator_learning_rate'
        )
        self.discriminator_learning_rate = tf.placeholder(
            tf.float32, [],
            name='discriminator_learning_rate'
        )

    def sample_encoded_context(self, embeddings):
        '''Helper function for init_opt'''
        if TYPE_KL_LOSS == 0:
            c = self.model.generate_condition(embeddings)
            # mean, var = tf.nn.moments(c, axes=[0, 1])
            # c = (c - mean) / tf.sqrt(var)
            # mean2, var2 = tf.nn.moments(c, axes=[0, 1])
            return c  # , mean, var, mean2, var2
        elif TYPE_KL_LOSS == 1:
            c_mean_logsigma = self.model.generate_gaussian_condition(embeddings)
            mean = c_mean_logsigma[0]
            epsilon = tf.random_normal(tf.shape(mean))
            stddev = tf.exp(c_mean_logsigma[1])
            c = mean + stddev * epsilon

            kl_loss = KL_loss(c_mean_logsigma[0], c_mean_logsigma[1])
            return c, kl_loss
        elif TYPE_KL_LOSS == 2:
            c = self.model.generate_condition(embeddings)
            mean, var = tf.nn.moments(c, axes=[0, 1])
            log_sigma = tf.log(tf.sqrt(var))
            kl_loss = KL_loss(mean, log_sigma)
            return c, kl_loss, mean, log_sigma
        else:
            NotImplementedError

    def get_interp_embeddings(self, embeddings):
        split0, split1 = tf.split(0, 2, embeddings)
        interp = tf.add(split0, split1) / 2.
        return tf.concat(0, [embeddings, interp])

    def init_opt(self):
        # self.images, self.masks, self.embeddings, self.bg_images
        self.build_placeholder()
        #
        # masks is tf.float32 with 0s and 1s
        if B_MASKED:
            real_images = tf.mul(self.images, tf.expand_dims(self.masks, 3))
            real_wrong_images = tf.mul(self.wrong_images, tf.expand_dims(self.masks, 3))
        else:
            real_images = self.images
            real_wrong_images = self.wrong_images

        with pt.defaults_scope(phase=pt.Phase.train):
            # ####get output from G network####################################
            if TYPE_KL_LOSS == 0:
                c = self.sample_encoded_context(self.embeddings)
            elif TYPE_KL_LOSS == 1:
                c, _ = self.sample_encoded_context(self.embeddings)
            elif TYPE_KL_LOSS == 2:
                c, _, _, _ = self.sample_encoded_context(self.embeddings)
            if cfg.NOISE_TYPE == 'normal':
                z = tf.random_normal([self.batch_size, cfg.Z_DIM])
            else:
                z = tf.random_uniform([self.batch_size, cfg.Z_DIM], minval=-1, maxval=1)

            self.log_vars.append(("hist_c", c))
            self.log_vars.append(("hist_z", z))
            fake_images = self.model.get_generator(tf.concat(1, [c, z]))

            ####################################################################
            interp_embeddings = self.get_interp_embeddings(self.embeddings)
            if TYPE_KL_LOSS == 0:
                interp_c = self.sample_encoded_context(interp_embeddings)
                # self.log_vars.append(("g_c_mean1", mean1))
                # self.log_vars.append(("g_c_var1", var1))
                # self.log_vars.append(("g_c_mean2", mean2))
                # self.log_vars.append(("g_c_var2", var2))
            elif TYPE_KL_LOSS == 1:
                interp_c, interp_kl_loss = self.sample_encoded_context(interp_embeddings)
            elif TYPE_KL_LOSS == 2:
                interp_c, interp_kl_loss, mean, log_sigma = self.sample_encoded_context(interp_embeddings)
                self.log_vars.append(("g_c_mean1", mean))
                self.log_vars.append(("g_c_log_sigma", log_sigma))
            if cfg.NOISE_TYPE == 'normal':
                interp_z = tf.random_normal([int(self.batch_size * 3 / 2), cfg.Z_DIM])
            else:
                interp_z = tf.random_uniform([int(self.batch_size * 3 / 2), cfg.Z_DIM], minval=-1, maxval=1)
            self.log_vars.append(("hist_interp_c", interp_c))
            self.log_vars.append(("hist_interp_z", interp_z))
            interp_fake_images = self.model.get_generator(tf.concat(1, [interp_c, interp_z]))

            ####################################################################
            if cfg.NOISE_TYPE == 'normal':
                noise_c = tf.random_normal([self.batch_size, cfg.GAN.EMBEDDING_DIM])
                noise_z = tf.random_normal([self.batch_size, cfg.Z_DIM])
            else:
                noise_c = tf.random_uniform([self.batch_size, cfg.GAN.EMBEDDING_DIM], minval=-1, maxval=1)
                noise_z = tf.random_uniform([self.batch_size, cfg.Z_DIM], minval=-1, maxval=1)
            self.log_vars.append(("hist_noise_c", noise_c))
            self.log_vars.append(("hist_noise_z", noise_z))
            noise_images = self.model.get_generator(tf.concat(1, [noise_c, noise_z]))

            # ####get discriminator_loss #######################################
            discriminator_loss_pre, discriminator_loss_cond = \
                self.compute_d_loss(real_images, real_wrong_images,
                                    fake_images, noise_images, c)
            if B_MI_LOSS > 0:
                d_mi = self.computeMI(c, fake_images, 1)
                self.log_vars.append(("d_mi", B_MI_LOSS * d_mi))

                d_mi_real = self.computeMI(c, real_images, 1)
                self.log_vars.append(("d_mi_real", B_MI_LOSS * d_mi_real))
                # discriminator_loss -= B_MI_LOSS * (d_mi + d_mi_real)
            if B_PRETRAIN:
                self.log_vars.append(("d_loss_total", discriminator_loss_pre))
            else:
                self.log_vars.append(("d_loss_total", discriminator_loss_cond))

            # ####get generator_loss ##########################################
            generator_loss_pre, generator_loss_cond =\
                self.compute_g_loss(interp_fake_images, noise_images, interp_c)
            # if TYPE_KL_LOSS > 0:
            #    generator_loss += interp_kl_loss
            #    self.log_vars.append(("g_interp_kl_loss", interp_kl_loss))
            if B_MI_LOSS:
                g_mi = self.computeMI(interp_c, interp_fake_images, 1)
                # generator_loss -= B_MI_LOSS * g_mi
                self.log_vars.append(("g_mi", B_MI_LOSS * g_mi))
            if B_PRETRAIN:
                self.log_vars.append(("g_loss_total", generator_loss_pre))
            else:
                # like_loss = self.compute_color_like_loss(real_images,
                #                                         fake_images, self.masks, 50)
                # generator_loss_cond += like_loss
                self.log_vars.append(("g_loss_total", generator_loss_cond))

            self.prepare_trainer(generator_loss_pre, discriminator_loss_pre,
                                 generator_loss_cond, discriminator_loss_cond)
            # #######define self.g_sum, self.d_sum,....########################
            self.define_summaries()

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("model", reuse=True) as scope:
                self.visualization(real_images)
                print("success")

    # ####get discriminator_loss and generator_loss for FG#####################
    def compute_d_loss(self, real_images, wrong_images, fake_images, noise_images, c):
        real_d = self.model.get_discriminator(real_images, self.embeddings)
        # real_d = self.model.get_noise_discriminator(real_images)
        real_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_d, tf.ones_like(real_d)))
        #
        wrong_d = self.model.get_discriminator(wrong_images, self.embeddings)
        # wrong_d = self.model.get_noise_discriminator(wrong_images)
        wrong_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(wrong_d, tf.zeros_like(wrong_d)))
        #
        fake_d = self.model.get_discriminator(fake_images, self.embeddings)  # c
        # fake_d = self.model.get_noise_discriminator(fake_images)
        fake_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_d, tf.zeros_like(fake_d)))
        if B_PRETRAIN == 0:
            self.log_vars.append(("d_loss_real", real_d_loss))
            self.log_vars.append(("d_loss_wrong", wrong_d_loss))
            self.log_vars.append(("d_loss_fake", fake_d_loss))
        discriminator_loss_cond = real_d_loss + (wrong_d_loss + fake_d_loss) / 2.

        real_d2 = self.model.get_noise_discriminator(real_images)
        real_d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_d2, tf.ones_like(real_d2)))
        #
        noise_d = self.model.get_noise_discriminator(noise_images)
        noise_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(noise_d, tf.zeros_like(noise_d)))
        if B_PRETRAIN == 1:
            self.log_vars.append(("d_loss_real2", real_d_loss2))
            self.log_vars.append(("d_loss_noise", noise_d_loss))
        discriminator_loss_pre = real_d_loss2 + noise_d_loss

        return discriminator_loss_pre, discriminator_loss_cond

    def compute_mean_covariance_with_mask(self, img, mask):
        shape = img.get_shape()
        batch_size = shape[0].value
        # height = shape[1].value
        # width = shape[2].value
        channel_num = shape[3].value

        mask = tf.expand_dims(mask, 3)
        fg_img = tf.mul(img, mask)
        fg_num_pixels = tf.reduce_sum(mask, [1, 2, 3], True)

        mu = tf.reduce_sum(fg_img, [1, 2], True)
        mu = tf.truediv(mu, fg_num_pixels)

        # shape is batch_size * num_pixels * channel_num
        img_hat = fg_img - mu
        fg_img_hat = tf.mul(img_hat, mask)
        fg_img_hat = tf.reshape(fg_img_hat, [batch_size, -1, channel_num])

        # shape is batch_size * channel_num * num_pixels
        fg_img_hat_transpose = tf.transpose(fg_img_hat, perm=[0, 2, 1])
        # shape is batch_size * channle_num * channle_num
        covariance = tf.batch_matmul(fg_img_hat_transpose, fg_img_hat)
        covariance = tf.truediv(tf.expand_dims(covariance, 3), fg_num_pixels)

        return mu, covariance

    def compute_color_like_loss(self, real_images, fake_images, masks, lamda):
        like_loss = tf.constant(0.)
        # fake_mu, fake_covariance = compute_mean_covariance(self.fake_images)
        # real_mu, real_covariance = compute_mean_covariance(self.fg_images)

        fake_mu, fake_covariance = self.compute_mean_covariance_with_mask(fake_images, masks)
        self.log_vars.append(("hist_fake_covariance", fake_covariance))
        self.log_vars.append(("hist_fake_mu", fake_mu))
        real_mu, real_covariance = self.compute_mean_covariance_with_mask(real_images, masks)
        self.log_vars.append(("hist_real_covariance", real_covariance))
        self.log_vars.append(("hist_real_mu", real_mu))

        like_loss_mu = lamda * tf.reduce_mean(tf.square(fake_mu - real_mu)) / 2.
        self.log_vars.append(("g_like_loss_mu", like_loss_mu))
        like_loss_covariance = 5 * lamda * tf.reduce_mean(tf.square(fake_covariance - real_covariance)) / 2.
        self.log_vars.append(("g_like_loss_covariance", like_loss_covariance))

        like_loss = like_loss_mu + like_loss_covariance
        self.log_vars.append(("g_like_loss", like_loss))
        return like_loss

    def compute_like_loss(self, real_images, fake_images):
        like_loss = tf.constant(0.)

        fake_L2_ftr = self.model.d_L2_template.construct(input=fake_images)
        real_L2_ftr = self.model.d_L2_template.construct(input=real_images)
        # like_loss_L2 = tf.reduce_mean(tf.square(fake_L2_ftr - real_L2_ftr)) / 2.
        # self.log_vars.append(("g_like_loss_L2", like_loss_L2))
        # like_loss += like_loss_L2

        fake_L3_ftr = self.model.d_L3_template.construct(L2=fake_L2_ftr)
        real_L3_ftr = self.model.d_L3_template.construct(L2=real_L2_ftr)
        like_loss_L3 = tf.reduce_mean(tf.square(fake_L3_ftr - real_L3_ftr)) / 2.
        self.log_vars.append(("g_like_loss_L3", 2 * like_loss_L3))
        like_loss += 2 * like_loss_L3

        fake_L4_sub1_ftr = self.model.d_L4_sub1_template.construct(L3=fake_L3_ftr)
        real_L4_sub1_ftr = self.model.d_L4_sub1_template.construct(L3=real_L3_ftr)
        like_loss_L4_sub1 = tf.reduce_mean(tf.square(fake_L4_sub1_ftr - real_L4_sub1_ftr)) / 2.
        self.log_vars.append(("g_like_loss_L4_sub1", 0.2 * like_loss_L4_sub1))
        like_loss += 0.2 * like_loss_L4_sub1

        fake_L4_sub2_ftr = self.model.d_L4_sub1_template.construct(L3=fake_L3_ftr)
        real_L4_sub2_ftr = self.model.d_L4_sub1_template.construct(L3=real_L3_ftr)
        like_loss_L4_sub2 = tf.reduce_mean(tf.square(fake_L4_sub2_ftr - real_L4_sub2_ftr)) / 2.
        self.log_vars.append(("g_like_loss_L4_sub2", 0.2 * like_loss_L4_sub2))
        like_loss += 0.2 * like_loss_L4_sub2

        return like_loss

    def computeMI(self, c, x_var, flag=1):
        '''Helper function for init_opt'''
        re_mean_logsigma = self.model.reconstuct_context(x_var)
        mean = re_mean_logsigma[0]
        stddev = tf.exp(re_mean_logsigma[1])
        epsilon = (c - mean) / (stddev + TINY)

        log_q_c_given_x = tf.reduce_mean(
            - 0.5 * np.log(2 * np.pi) - tf.log(stddev + TINY) - 0.5 * tf.square(epsilon),
        )
        cross_entropy = - log_q_c_given_x
        if flag == 0:
            log_p_c = tf.reduce_mean(
                - 0.5 * np.log(2 * np.pi) - np.log(1) - 0.5 * tf.square(c)
            )
            prior_entropy = - log_p_c
        elif flag == 1:
            prior_entropy = 0.5 + 0.5 * np.log(2 * np.pi)
        else:
            raise NotImplementedError
        mutual_info = prior_entropy - cross_entropy
        return mutual_info

    def compute_g_loss(self, fake_images, noise_images, c):
        fake_g = self.model.get_discriminator(fake_images, self.interp_embeddings)  # self.interp_embeddings
        # fake_g = self.model.get_noise_discriminator(fake_images)
        fake_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_g, tf.ones_like(fake_g)))
        if B_PRETRAIN == 0:
            self.log_vars.append(("g_loss_fake", fake_g_loss))
        generator_loss_cond = fake_g_loss

        #
        noise_g = self.model.get_noise_discriminator(noise_images)
        noise_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(noise_g, tf.ones_like(noise_g)))
        if B_PRETRAIN == 1:
            self.log_vars.append(("g_loss_noise", noise_g_loss))
        generator_loss_pre = noise_g_loss

        return generator_loss_pre, generator_loss_cond

    def prepare_trainer(self, generator_loss_pre, discriminator_loss_pre,
                        generator_loss_cond, discriminator_loss_cond):
        '''Helper function for init_opt'''
        all_vars = tf.trainable_variables()

        g_vars = [var for var in all_vars if
                  var.name.startswith('g_')]
        d_vars = [var for var in all_vars if
                  var.name.startswith('d_')]

        generator_optimizer = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=0.5)
        self.generator_trainer_pre = pt.apply_optimizer(generator_optimizer,
                                                        losses=[generator_loss_pre],
                                                        var_list=g_vars)
        self.generator_trainer_cond = pt.apply_optimizer(generator_optimizer,
                                                         losses=[generator_loss_cond],
                                                         var_list=g_vars)
        discriminator_optimizer = tf.train.AdamOptimizer(self.discriminator_learning_rate, beta1=0.5)
        self.discriminator_trainer_pre = pt.apply_optimizer(discriminator_optimizer,
                                                            losses=[discriminator_loss_pre],
                                                            var_list=d_vars)
        self.discriminator_trainer_cond = pt.apply_optimizer(discriminator_optimizer,
                                                             losses=[discriminator_loss_cond],
                                                             var_list=d_vars)
        self.log_vars.append(("g_learning_rate", self.generator_learning_rate))
        self.log_vars.append(("d_learning_rate", self.discriminator_learning_rate))

    def define_summaries(self):
        '''Helper function for init_opt'''
        all_sum = {'g': [], 'd': [], 'hist': []}
        for k, v in self.log_vars:
            if k.startswith('g'):
                all_sum['g'].append(tf.scalar_summary(k, v))
            elif k.startswith('d'):
                all_sum['d'].append(tf.scalar_summary(k, v))
            elif k.startswith('hist'):
                all_sum['hist'].append(tf.histogram_summary(k, v))

        self.g_sum = tf.merge_summary(all_sum['g'])
        self.d_sum = tf.merge_summary(all_sum['d'])
        self.hist_sum = tf.merge_summary(all_sum['hist'])

    def visualize_one_superimage(self, img_var, images, rows, filename):
        stacked_img = []
        for row in range(rows):
            img = images[row * rows, :, :, :]
            row_img = [img]  # real image and masked images
            for col in range(rows):
                row_img.append(img_var[row * rows + col, :, :, :])
            # each rows is 1realimage +10_fakeimage
            stacked_img.append(tf.concat(1, row_img))
        imgs = tf.expand_dims(tf.concat(0, stacked_img), 0)
        current_img_summary = tf.image_summary(filename, imgs)
        return current_img_summary, imgs

    def visualization(self, real_images):
        if TYPE_KL_LOSS == 0:
            c = self.sample_encoded_context(self.embeddings)
        elif TYPE_KL_LOSS == 1:
            c, _ = self.sample_encoded_context(self.embeddings)
        elif TYPE_KL_LOSS == 2:
            c, _, _, _ = self.sample_encoded_context(self.embeddings)
        if cfg.NOISE_TYPE == 'normal':
            z = tf.random_normal([self.batch_size, cfg.Z_DIM])
        else:
            z = tf.random_uniform([self.batch_size, cfg.Z_DIM], minval=-1, maxval=1)
        fake_x = self.model.get_generator(tf.concat(1, [c, z]))
        fake_sum_train, superimage_train = self.visualize_one_superimage(fake_x[:64, :, :, :],
                                                                         real_images[:64, :, :, :],
                                                                         8, "train_on_text")
        fake_sum_test, superimage_test = self.visualize_one_superimage(fake_x[64:128, :, :, :],
                                                                       real_images[64:128, :, :, :],
                                                                       8, "test_on_text")
        if cfg.NOISE_TYPE == 'normal':
            noise_c = tf.random_normal([self.batch_size, cfg.GAN.EMBEDDING_DIM])
        else:
            noise_c = tf.random_uniform([self.batch_size, cfg.GAN.EMBEDDING_DIM], minval=-1, maxval=1)
        noise_x = self.model.get_generator(tf.concat(1, [noise_c, z]))
        noise_sum_train, _ = self.visualize_one_superimage(noise_x[:64, :, :, :],
                                                           real_images[:64, :, :, :],
                                                           8, "train_on_noise")
        self.superimages = tf.concat(0, [superimage_train, superimage_test])
        self.image_summary = tf.merge_summary([fake_sum_train, fake_sum_test, noise_sum_train])

    def preprocess(self, x):
        # make sure every row with 10 column have the same embeddings
        for i in range(8):
            for j in range(1, 8):
                x[i * 8 + j] = x[i * 8]
        return x

    def epoch_sum_images(self, sess):
        images_train, _, masks_train, embeddings_train, captions_train, _, _ = self.dataset.train.next_batch(64, 1)
        images_train = self.preprocess(images_train)
        masks_train = self.preprocess(masks_train)
        embeddings_train = self.preprocess(embeddings_train)

        images_test, _, masks_test, embeddings_test, captions_test, _, _ = self.dataset.test.next_batch(64, 1)
        images_test = self.preprocess(images_test)
        masks_test = self.preprocess(masks_test)
        embeddings_test = self.preprocess(embeddings_test)

        images = np.concatenate([images_train, images_test], axis=0)
        masks = np.concatenate([masks_train, masks_test], axis=0)
        embeddings = np.concatenate([embeddings_train, embeddings_test], axis=0)

        if self.batch_size > 128:
            images_pad, _, masks_pad, embeddings_pad, _, _, _ = self.dataset.test.next_batch(self.batch_size - 128, 1)
            images = np.concatenate([images, images_pad], axis=0)
            masks = np.concatenate([masks, masks_pad], axis=0)
            embeddings = np.concatenate([embeddings, embeddings_pad], axis=0)
        feed_dict = {self.images: images,
                     self.masks: masks,
                     self.embeddings: embeddings
                     }
        gen_samples, img_summary = sess.run([self.superimages, self.image_summary], feed_dict)

        # save images generated for train and test captions
        scipy.misc.imsave('%s/train.jpg' % (self.log_dir), gen_samples[0])
        pfi_train = open(self.log_dir + "/train.txt", "w")

        scipy.misc.imsave('%s/test.jpg' % (self.log_dir), gen_samples[1])
        pfi_test = open(self.log_dir + "/test.txt", "w")
        for row in range(8):
            pfi_train.write('\n***row %d***\n' % row)
            pfi_train.write(captions_train[row * 8])

            pfi_test.write('\n***row %d***\n' % row)
            pfi_test.write(captions_train[row * 8])
        pfi_train.close()
        pfi_test.close()

        return img_summary

    def train(self):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                self.init_opt()

                saver = tf.train.Saver(tf.all_variables(), keep_checkpoint_every_n_hours=2)
                if len(self.model_path) > 0:
                    print("Reading model parameters from %s" % self.model_path)
                    saver.restore(sess, self.model_path)
                    counter = self.model_path[self.model_path.rfind('_') + 1:self.model_path.rfind('.')]
                    counter = int(counter)
                    if not cfg.TRAIN.FLAG:
                        self.epoch_save_samples(sess, cfg.TRAIN.NUM_COPY)
                        return
                else:
                    print("Created model with fresh parameters.")
                    sess.run(tf.initialize_all_variables())
                    counter = 0

                # summary_op = tf.merge_all_summaries()
                summary_writer = tf.train.SummaryWriter(self.log_dir, sess.graph)

                keys = ["d_loss", "g_loss"]
                log_vars = []
                log_keys = []
                for k, v in self.log_vars:
                    if k in keys:
                        log_vars.append(v)
                        log_keys.append(k)
                        # print(k, v)

                updates_per_epoch = int(self.dataset.train._num_examples / self.batch_size)
                generator_learning_rate = cfg.TRAIN.GENERATOR_LR
                discriminator_learning_rate = cfg.TRAIN.DISCRIMINATOR_LR
                for epoch in range(int(counter / updates_per_epoch), self.max_epoch):
                    widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(maxval=updates_per_epoch, widgets=widgets)
                    pbar.start()

                    if epoch % 30 == 0 and epoch != 0 and generator_learning_rate > 0.000001:
                        generator_learning_rate *= 0.5  # TODO:0.5; 0.2
                        discriminator_learning_rate *= 0.5
                    #if generator_learning_rate > 0.000001:
                    #    generator_learning_rate = generator_learning_rate * 0.97
                    #    discriminator_learning_rate = discriminator_learning_rate * 0.95

                    all_log_vals = []
                    if B_PRETRAIN:
                        discriminator_trainer = self.discriminator_trainer_pre
                        generator_trainer = self.generator_trainer_pre
                    else:
                        discriminator_trainer = self.discriminator_trainer_cond
                        generator_trainer = self.generator_trainer_cond

                    for i in range(updates_per_epoch):
                        pbar.update(i)
                        # Prepare a batch of data
                        images, wrong_images, masks, embeddings, _, _, _ = self.dataset.train.next_batch(self.batch_size, 4)
                        feed_dict = {self.images: images,
                                     self.masks: masks.astype(np.float32),
                                     self.wrong_images: wrong_images,
                                     self.embeddings: embeddings,
                                     self.generator_learning_rate: generator_learning_rate,
                                     self.discriminator_learning_rate: discriminator_learning_rate
                                     }
                        # train d
                        feed_out = [discriminator_trainer,
                                    self.d_sum,
                                    self.hist_sum,
                                    log_vars,
                                    ]
                        _, d_summary, hist_summary, log_vals = sess.run(feed_out, feed_dict)
                        summary_writer.add_summary(d_summary, counter)
                        summary_writer.add_summary(hist_summary, counter)
                        all_log_vals.append(log_vals)
                        # train g
                        feed_out = [generator_trainer,
                                    self.g_sum,
                                    ]
                        _, g_summary = sess.run(
                            feed_out, feed_dict
                        )
                        summary_writer.add_summary(g_summary, counter)
                        # save checkpoint
                        counter += 1
                        if counter % self.snapshot_interval == 0:
                            snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                            fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                            print("Model saved in file: %s" % fn)

                    summary_writer.add_summary(self.epoch_sum_images(sess), counter)

                    avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                    dic_logs = {}
                    for k, v in zip(log_keys, avg_log_vals):
                        dic_logs[k] = v
                        # print(k, v)

                    log_line = "; ".join("%s: %s" % (str(k), str(dic_logs[k])) for k in dic_logs)  # zip(log_keys, avg_log_vals)
                    print("Epoch %d | " % (epoch) + log_line)
                    sys.stdout.flush()
                    if np.any(np.isnan(avg_log_vals)):
                        raise ValueError("NaN detected!")
