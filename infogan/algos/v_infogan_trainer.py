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
        self.updates_per_epoch = cfg.TRAIN.UPDATES_PER_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.model_path = cfg.TRAIN.PRETRAINED_MODEL

        self.generator_learning_rate = cfg.TRAIN.GENERATOR_LR
        self.discriminator_learning_rate = cfg.TRAIN.DISCRIMINATOR_LR
        self.bg_discriminator_learning_rate = cfg.TRAIN.BG_DISCRIMINATOR_LR
        self.fg_discriminator_learning_rate = cfg.TRAIN.FG_DISCRIMINATOR_LR
        self.encoder_learning_rate = cfg.TRAIN.ENCODER_LR

        self.discriminator_trainer = None
        self.generator_trainer = None
        self.images = None
        self.masks = None
        self.embeddings = None
        self.bg_images = None
        self.fg_images = None
        self.log_vars = []

    def build_placeholder(self):
        '''Helper function for init_opt'''
        self.images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='real_images')
        self.masks = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape[:2],
            name='real_masks')
        self.embeddings = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.embedding_shape,
            name='conditional_embeddings'
        )
        self.bg_images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='real_bg_images')

    def sample_encoded_context(self, bencode=True):
        '''Helper function for init_opt'''
        if bencode:
            c_mean_logsigma = self.model.generate_condition(self.embeddings)
            self.log_vars.append(("hist_c_mean", c_mean_logsigma[0]))
            self.log_vars.append(("hist_c_log_sigma", c_mean_logsigma[0]))
            c_sample = self.model.con_latent_dist.sample_prior(self.batch_size)
            c = c_mean_logsigma[0] + c_mean_logsigma[1] * c_sample

            kl_loss = KL_loss(c_mean_logsigma[0], c_mean_logsigma[1])
        else:
            c = self.embeddings
            kl_loss = 0
        return c, kl_loss

    def init_opt(self):
        # self.images, self.masks, self.embeddings, self.bg_images
        self.build_placeholder()
        # masks is tf.float32 with 0s and 1s
        self.fg_images = tf.mul(self.images, tf.expand_dims(self.masks, 3))

        with pt.defaults_scope(phase=pt.Phase.train):
            # ####prepare input for G #########################################
            # A fixed set of z for generating BG in fake_x and noise_x
            z = self.model.latent_dist.sample_prior(self.batch_size)
            reg_z = self.model.reg_z(z)
            self.log_vars.append(("hist_reg_z", reg_z))
            # zeros z for generating fG in fg_x
            fg_z = tf.zeros_like(z)
            self.log_vars.append(("hist_fg_z", fg_z))
            # c sampled from encoded text for generating FG in fake_x
            c, kl_loss = self.sample_encoded_context(bencode=True)
            self.log_vars.append(("hist_fake_c", c))
            self.log_vars.append(("e_kl_loss", kl_loss))
            # c randomly samples from Gaussian for generating FG in noise_x
            noise_c = self.model.con_latent_dist.sample_prior(self.batch_size)
            self.log_vars.append(("hist_noise_c", noise_c))
            # c with all 0s for generating BG in bg_x
            bg_c = tf.zeros_like(noise_c)
            self.log_vars.append(("hist_bg_c", bg_c))

            # ####get output from G network####################################
            self.fake_x = self.model.get_generator(tf.concat(1, [c, z]))
            self.noise_x = self.model.get_generator(tf.concat(1, [noise_c, z]))
            self.bg_x = self.model.get_generator(tf.concat(1, [bg_c, z]))
            self.fg_x = self.model.get_generator(tf.concat(1, [c, fg_z]))

            # ####get discriminator_loss and generator_loss####################
            # from D (real image detector)#####################################
            discriminator_loss, generator_loss = self.compute_loss_from_D()

            # #####Like loss###################################################
            if cfg.TRAIN.MASK_FLAG and cfg.TRAIN.COEFF.LIKE > TINY:
                like_loss = self.compute_like_loss_from_D()
                encoder_loss = kl_loss + like_loss
                # ##Add like loss to ......
                generator_loss += cfg.TRAIN.COEFF.LIKE * like_loss
                # discriminator_loss += cfg.TRAIN.COEFF.LIKE * like_loss  # TODO: whether add this ??
                self.log_vars.append(("g_d_like_loss_reweight", cfg.TRAIN.COEFF.LIKE * like_loss))

            # ####get discriminator_loss and generator_loss for BG##############
            # ####Different discriminators but the same generator###############
            bg_discriminator_loss, bg_generator_loss = self.compute_loss_from_bg_D(reg_z)
            generator_loss += bg_generator_loss  # For shared generator

            # ####get discriminator_loss and generator_loss for FG##############
            # ####Different discriminators but the same generator###############
            fg_discriminator_loss, fg_generator_loss = self.compute_loss_from_fg_D()
            generator_loss += fg_generator_loss  # For shared generator
            if cfg.TRAIN.MASK_FLAG and cfg.TRAIN.COEFF.LIKE > TINY:
                fg_like_loss = self.compute_like_loss_from_fg_D()
                encoder_loss += fg_like_loss
                # ##Add like loss to ......
                generator_loss += cfg.TRAIN.COEFF.LIKE * fg_like_loss
                # discriminator_loss += cfg.TRAIN.COEFF.LIKE * fg_like_loss  # TODO: whether add this ??

            # #######Total loss for build#######################################
            # #######self.encoder_trainer, self.generator_trainer ##############
            # ######self.discriminator_trainer, self.bg_discriminator_trainer###
            self.log_vars.append(("e_loss", encoder_loss))
            self.log_vars.append(("g_loss", generator_loss))
            self.log_vars.append(("d_loss", discriminator_loss))
            self.log_vars.append(("bg_d_loss", bg_discriminator_loss))
            self.log_vars.append(("fg_d_loss", fg_discriminator_loss))
            self.prepare_trainer(encoder_loss, generator_loss,
                                 discriminator_loss, bg_discriminator_loss,
                                 fg_discriminator_loss)
            # #######define self.e_sum, self.g_sum, self.d_sum,################
            # #######self.bg_d_sum, self.fg_d_sum, self.hist_sum, self.other_sum##############
            self.define_summaries()

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("model", reuse=True) as scope:
                self.visualization()
                print("success")

    def computeMI(self, dist, reg_z, reg_dist_info, bprior):
        '''Helper function for init_opt'''
        # TODO: normalize by dimention ??
        log_q_c_given_x = dist.logli(reg_z, reg_dist_info)
        cross_ent = tf.reduce_mean(-log_q_c_given_x)
        if bprior:
            log_q_c = dist.logli_prior(reg_z)
            ent = tf.reduce_mean(-log_q_c)
            mi_est = ent - cross_ent
        else:
            mi_est = -cross_ent
        mi_est /= tf.to_float(dist.dim)
        return mi_est

    # #####Like loss###################################################
    def compute_like_loss_from_D(self):
        fake_fg_x = tf.mul(self.fake_x, tf.expand_dims(self.masks, 3))
        # ##like loss based on element-wise distance between real and fake images
        # like_loss = tf.reduce_mean(tf.square(self.fg_images - fake_fg_x)) / 2.

        # ##like loss based on feature distance between real and fake images
        real_fg_shared_layers = self.model.get_discriminator_shared(self.fg_images)
        fake_fg_shared_layers = self.model.get_discriminator_shared(fake_fg_x)
        like_loss = tf.reduce_mean(tf.square(real_fg_shared_layers - fake_fg_shared_layers)) / 2.
        self.log_vars.append(("e_like_loss", like_loss))
        return like_loss

    def compute_like_loss_from_fg_D(self):
        # ##like loss based on element-wise distance between real and fake images
        # like_loss = tf.reduce_mean(tf.square(self.fg_images - self.fg_x)) / 2.

        # ##like loss based on feature distance between real and fake images
        real_fg_shared_layers = self.model.get_fg_discriminator_shared(self.fg_images)
        fg_shared_layers = self.model.get_fg_discriminator_shared(self.fg_x)
        like_loss = tf.reduce_mean(tf.square(real_fg_shared_layers - fg_shared_layers)) / 2.
        self.log_vars.append(("e_fg_like_loss", like_loss))
        return like_loss

    # ####get discriminator_loss and generator_loss####################
    # from D (real image detector)#####################################
    def compute_loss_from_D(self):
        real_shared_layers = self.model.get_discriminator_shared(self.images)
        real_bg_shared_layers = self.model.get_discriminator_shared(self.bg_images)
        real_fg_shared_layers = self.model.get_discriminator_shared(self.fg_images)
        fake_shared_layers = self.model.get_discriminator_shared(self.fake_x)
        noise_shared_layers = self.model.get_discriminator_shared(self.noise_x)
        bg_shared_layers = self.model.get_discriminator_shared(self.bg_x)
        fg_shared_layers = self.model.get_discriminator_shared(self.fg_x)

        real_d = self.model.get_discriminator(real_shared_layers)
        real_bg_d = self.model.get_discriminator(real_bg_shared_layers)
        real_fg_d = self.model.get_discriminator(real_fg_shared_layers)
        fake_d = self.model.get_discriminator(fake_shared_layers)
        noise_d = self.model.get_discriminator(noise_shared_layers)
        bg_d = self.model.get_discriminator(bg_shared_layers)
        fg_d = self.model.get_discriminator(fg_shared_layers)
        # Only images with both BG and FG are real images, all others are fake images
        d_loss_legit = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_d, tf.ones_like(real_d)))
        d_loss_bg_legit = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_bg_d, tf.zeros_like(real_bg_d)))
        d_loss_fg_legit = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_fg_d, tf.zeros_like(real_fg_d)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_d, tf.zeros_like(fake_d)))
        d_loss_noise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(noise_d, tf.zeros_like(noise_d)))
        d_loss_bg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(bg_d, tf.zeros_like(bg_d)))
        d_loss_fg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fg_d, tf.zeros_like(fg_d)))

        discriminator_loss = d_loss_legit + (d_loss_fake + d_loss_noise) / 2.
        discriminator_loss += (d_loss_bg_legit + d_loss_fg_legit + d_loss_bg + d_loss_fg) / 4.
        self.log_vars.append(("d_loss_real", d_loss_legit))
        self.log_vars.append(("d_loss_real_bg", d_loss_bg_legit))
        self.log_vars.append(("d_loss_real_fg", d_loss_fg_legit))
        self.log_vars.append(("d_loss_fake", d_loss_fake))
        self.log_vars.append(("d_loss_noise", d_loss_noise))
        self.log_vars.append(("d_loss_bg", d_loss_bg))
        self.log_vars.append(("d_loss_fg", d_loss_fg))
        # real_p = tf.nn.sigmoid(real_d)
        # feak_p = tf.nn.sigmoid(fake_d)
        # self.log_vars.append(("max_real_p", tf.reduce_max(real_p)))
        # self.log_vars.append(("min_real_p", tf.reduce_min(real_p)))
        # self.log_vars.append(("max_fake_p", tf.reduce_max(feak_p)))
        # self.log_vars.append(("min_fake_p", tf.reduce_min(feak_p)))

        # Train genarator to generate images as real as possible
        g_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_d, tf.ones_like(fake_d)))
        g_loss_noise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(noise_d, tf.ones_like(noise_d)))
        # TODO: change the ratio of these two;
        # TODO: add d_loss_bg to train generator to avoid generating BG images
        generator_loss = (g_loss_fake + g_loss_noise) / 2.
        self.log_vars.append(("g_loss_fake", g_loss_fake))
        self.log_vars.append(("g_loss_noise", g_loss_noise))

        return discriminator_loss, generator_loss

    # ####get discriminator_loss and generator_loss for BG#####################
    def compute_loss_from_bg_D(self, reg_z):
        real_shared_layers = self.model.get_bg_discriminator_shared(self.images)
        real_bg_shared_layers = self.model.get_bg_discriminator_shared(self.bg_images)
        real_fg_shared_layers = self.model.get_fg_discriminator_shared(self.fg_images)
        fake_shared_layers = self.model.get_bg_discriminator_shared(self.fake_x)
        noise_shared_layers = self.model.get_bg_discriminator_shared(self.noise_x)
        bg_shared_layers = self.model.get_bg_discriminator_shared(self.bg_x)
        fg_shared_layers = self.model.get_bg_discriminator_shared(self.fg_x)

        real_d = self.model.get_bg_discriminator(real_shared_layers)
        real_bg_d = self.model.get_bg_discriminator(real_bg_shared_layers)
        real_fg_d = self.model.get_bg_discriminator(real_fg_shared_layers)
        fake_d = self.model.get_bg_discriminator(fake_shared_layers)
        noise_d = self.model.get_bg_discriminator(noise_shared_layers)
        bg_d = self.model.get_bg_discriminator(bg_shared_layers)
        fg_d = self.model.get_bg_discriminator(fg_shared_layers)

        # Only real BG images are real images, all others are fake images
        d_loss_bg_legit = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_bg_d, tf.ones_like(real_bg_d)))
        d_loss_fg_legit = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_fg_d, tf.zeros_like(real_fg_d)))
        d_loss_legit = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_d, tf.zeros_like(real_d)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_d, tf.zeros_like(fake_d)))
        d_loss_noise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(noise_d, tf.zeros_like(noise_d)))
        d_loss_bg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(bg_d, tf.zeros_like(bg_d)))
        d_loss_fg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fg_d, tf.zeros_like(fg_d)))

        # TODO: add d_loss_fake2 and d_loss_noise2 to train generator to avoid generating images with FGs
        bg_discriminator_loss = d_loss_bg_legit + d_loss_bg
        bg_discriminator_loss += (d_loss_fg_legit + d_loss_legit + d_loss_fake + d_loss_noise + d_loss_fg) / 5.
        self.log_vars.append(("bg_d_loss_real", d_loss_legit))
        self.log_vars.append(("bg_d_loss_real_bg", d_loss_bg_legit))
        self.log_vars.append(("bg_d_loss_real_fg", d_loss_fg_legit))
        self.log_vars.append(("bg_d_loss_fake", d_loss_fake))
        self.log_vars.append(("bg_d_loss_noise", d_loss_noise))
        self.log_vars.append(("bg_d_loss_bg", d_loss_bg))
        self.log_vars.append(("bg_d_loss_fg", d_loss_fg))

        bg_generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(bg_d, tf.ones_like(bg_d)))
        self.log_vars.append(("g_bg_loss", bg_generator_loss))

        # #######MI between fake_bg and the prior c/noise/reg_z#############
        mi_sum = tf.constant(0.)
        # ###Reconstruct reg_z  from bg_x
        if cfg.TRAIN.COEFF.REG_Z > TINY:
            bg_reg_z_dist_info = self.model.reconstuct_reg(bg_shared_layers)
            bg_reg_z_mi = self.computeMI(self.model.reg_latent_dist, reg_z, bg_reg_z_dist_info, 1)
            mi_sum += cfg.TRAIN.COEFF.FAKE_REG_Z * bg_reg_z_mi
            self.log_vars.append(("MI_bg_reg_z", bg_reg_z_mi))
        bg_discriminator_loss -= mi_sum
        bg_generator_loss -= mi_sum
        # self.log_vars.append(("MI", mi_sum))

        return bg_discriminator_loss, bg_generator_loss

    # ####get discriminator_loss and generator_loss for FG#####################
    def compute_loss_from_fg_D(self):
        real_shared_layers = self.model.get_fg_discriminator_shared(self.images)
        real_bg_shared_layers = self.model.get_fg_discriminator_shared(self.bg_images)
        real_fg_shared_layers = self.model.get_fg_discriminator_shared(self.fg_images)
        fake_shared_layers = self.model.get_fg_discriminator_shared(self.fake_x)
        noise_shared_layers = self.model.get_fg_discriminator_shared(self.noise_x)
        bg_shared_layers = self.model.get_fg_discriminator_shared(self.bg_x)
        fg_shared_layers = self.model.get_fg_discriminator_shared(self.fg_x)

        real_d = self.model.get_fg_discriminator(real_shared_layers)
        real_bg_d = self.model.get_fg_discriminator(real_bg_shared_layers)
        real_fg_d = self.model.get_fg_discriminator(real_fg_shared_layers)
        fake_d = self.model.get_fg_discriminator(fake_shared_layers)
        noise_d = self.model.get_fg_discriminator(noise_shared_layers)
        bg_d = self.model.get_fg_discriminator(bg_shared_layers)
        fg_d = self.model.get_fg_discriminator(fg_shared_layers)

        # Only real FG images are real images, all others are fake images
        d_loss_fg_legit = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_fg_d, tf.ones_like(real_fg_d)))
        d_loss_bg_legit = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_bg_d, tf.zeros_like(real_bg_d)))
        d_loss_legit = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_d, tf.zeros_like(real_d)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_d, tf.zeros_like(fake_d)))
        d_loss_noise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(noise_d, tf.zeros_like(noise_d)))
        d_loss_bg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(bg_d, tf.zeros_like(bg_d)))
        d_loss_fg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fg_d, tf.zeros_like(fg_d)))

        # TODO: add d_loss_fake2 and d_loss_noise2 to train generator to avoid generating images with FGs
        fg_discriminator_loss = d_loss_fg_legit + d_loss_fg
        fg_discriminator_loss += (d_loss_bg_legit + d_loss_legit + d_loss_fake + d_loss_noise + d_loss_bg) / 5.
        self.log_vars.append(("fg_d_loss_real", d_loss_legit))
        self.log_vars.append(("fg_d_loss_real_fg", d_loss_fg_legit))
        self.log_vars.append(("fg_d_loss_real_bg", d_loss_bg_legit))
        self.log_vars.append(("fg_d_loss_fake", d_loss_fake))
        self.log_vars.append(("fg_d_loss_noise", d_loss_noise))
        self.log_vars.append(("fg_d_loss_bg", d_loss_bg))
        self.log_vars.append(("fg_d_loss_fg", d_loss_fg))

        fg_generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fg_d, tf.ones_like(fg_d)))
        self.log_vars.append(("g_fg_loss", fg_generator_loss))

        return fg_discriminator_loss, fg_generator_loss

    def prepare_trainer(self, encoder_loss, generator_loss, discriminator_loss,
                        bg_discriminator_loss, fg_discriminator_loss):
        '''Helper function for init_opt'''
        all_vars = tf.trainable_variables()

        e_vars = [var for var in all_vars if
                  var.name.startswith('e_')]
        g_vars = [var for var in all_vars if
                  var.name.startswith('g_')]
        d_vars = [var for var in all_vars if
                  var.name.startswith('d_')]
        r_vars = [var for var in all_vars if
                  var.name.startswith('r_')]
        bg_d_vars = [var for var in all_vars if
                     var.name.startswith('bg_')]
        fg_d_vars = [var for var in all_vars if
                     var.name.startswith('fg_')]

        encoder_optimizer = tf.train.AdamOptimizer(self.encoder_learning_rate, beta1=0.5)
        self.encoder_trainer = pt.apply_optimizer(encoder_optimizer,
                                                  losses=[encoder_loss],
                                                  var_list=e_vars)
        generator_optimizer = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=0.5)
        self.generator_trainer = pt.apply_optimizer(generator_optimizer,
                                                    losses=[generator_loss],
                                                    var_list=g_vars)
        discriminator_optimizer = tf.train.AdamOptimizer(self.discriminator_learning_rate, beta1=0.5)
        self.discriminator_trainer = pt.apply_optimizer(discriminator_optimizer,
                                                        losses=[discriminator_loss],
                                                        var_list=d_vars + r_vars)
        bg_discriminator_optimizer = tf.train.AdamOptimizer(self.bg_discriminator_learning_rate, beta1=0.5)
        self.bg_discriminator_trainer = pt.apply_optimizer(bg_discriminator_optimizer,
                                                           losses=[bg_discriminator_loss],
                                                           var_list=bg_d_vars)
        fg_discriminator_optimizer = tf.train.AdamOptimizer(self.fg_discriminator_learning_rate, beta1=0.5)
        self.fg_discriminator_trainer = pt.apply_optimizer(fg_discriminator_optimizer,
                                                           losses=[fg_discriminator_loss],
                                                           var_list=fg_d_vars)

    def define_summaries(self):
        '''Helper function for init_opt'''
        all_sum = {'e': [], 'g': [], 'd': [], 'bg': [], 'fg': [],
                   'hist': []}
        for k, v in self.log_vars:
            if k.startswith('e_'):
                all_sum['e'].append(tf.scalar_summary(k, v))
            elif k.startswith('g_'):
                all_sum['g'].append(tf.scalar_summary(k, v))
            elif k.startswith('d_'):
                all_sum['d'].append(tf.scalar_summary(k, v))
            elif k.startswith('bg_'):
                all_sum['bg'].append(tf.scalar_summary(k, v))
            elif k.startswith('fg_'):
                all_sum['fg'].append(tf.scalar_summary(k, v))
            elif k.startswith('hist_'):
                all_sum['hist'].append(tf.histogram_summary(k, v))
            # else:
            #    all_sum['others'].append(tf.scalar_summary(k, v))

        self.e_sum = tf.merge_summary(all_sum['e'])
        self.g_sum = tf.merge_summary(all_sum['g'])
        self.d_sum = tf.merge_summary(all_sum['d'])
        self.bg_d_sum = tf.merge_summary(all_sum['bg'])
        self.fg_d_sum = tf.merge_summary(all_sum['fg'])
        self.hist_sum = tf.merge_summary(all_sum['hist'])
        # self.other_sum = tf.merge_summary(all_sum['others'])

    def generate_bg_pad(self, bg_img, bg_mask, dim_indices):
        '''Helper function for extract_padded_bg_images'''
        nonzero_num = tf.reduce_sum(bg_mask, reduction_indices=dim_indices, keep_dims=True)
        value_sum = tf.reduce_sum(bg_img, reduction_indices=dim_indices, keep_dims=True)
        value_mean = tf.div(value_sum, nonzero_num)
        bg_pad = tf.mul(1. - bg_mask, tf.add(bg_img, value_mean))
        return bg_pad

    def extract_bg_images(self, images, masks, bpad=False):
        bg_masks = 1.0 - masks
        bg_imgs = tf.mul(images, bg_masks)
        if bpad:
            # bg_pad_mean = self.generate_bg_pad(bg_imgs, bg_masks, [1, 2])
            # bg_imgs = tf.add(bg_imgs, bg_pad_mean)
            bg_pad_rowmean = self.generate_bg_pad(bg_imgs, bg_masks, 1)
            bg_pad_colmean = self.generate_bg_pad(bg_imgs, bg_masks, 2)
            bg_imgs = tf.add(bg_imgs, (bg_pad_rowmean + bg_pad_colmean) / 2.)
        return bg_imgs

    def visualize_one_superimage(self, img_var, images, masks, bg_images, rows, filename):
        # bg_images = self.extract_bg_images(images, masks, bpad=False)
        fg_images = tf.mul(images, masks)
        stacked_img = []
        for row in range(rows):
            img = images[row * rows, :, :, :]
            masked_img = fg_images[row * rows, :, :, :]
            bg_img = bg_images[row * rows, :, :, :]
            row_img = [img, masked_img, bg_img]  # real image and masked images
            for col in range(rows):
                row_img.append(img_var[row * rows + col, :, :, :])
            # each rows is 1realimage +10_fakeimage
            stacked_img.append(tf.concat(1, row_img))
        imgs = tf.concat(0, stacked_img)
        imgs = tf.expand_dims(imgs, 0)
        current_img_summary = tf.image_summary(filename, imgs)
        return current_img_summary

    def visualization(self):
        # A fixed set of z for BG generation
        z_row = self.model.latent_dist.sample_prior(8)
        z = tf.tile(z_row, tf.constant([16, 1]))
        if self.batch_size > 128:
            z_pad = self.model.latent_dist.sample_prior(self.batch_size - 128)
            z = tf.concat(0, [z, z_pad])

        # c sampled from encoded text
        c, _ = self.sample_encoded_context(bencode=True)
        fake_x = self.model.get_generator(tf.concat(1, [c, z]))
        img_sum1 = self.visualize_one_superimage(fake_x[:64, :, :, :],
                                                 self.images[:64, :, :, :],
                                                 tf.expand_dims(self.masks[:64, :, :], 3),
                                                 self.bg_images[:64, :, :, :],
                                                 8, "train_image_on_text")
        img_sum2 = self.visualize_one_superimage(fake_x[64:128, :, :, :],
                                                 self.images[64:128, :, :, :],
                                                 tf.expand_dims(self.masks[64:128, :, :], 3),
                                                 self.bg_images[64:128, :, :, :],
                                                 8, "test_image_on_text")

        fg_z = tf.zeros_like(z)
        fg_x = self.model.get_generator(tf.concat(1, [c, fg_z]))
        img_sum3 = self.visualize_one_superimage(fg_x[:64, :, :, :],
                                                 self.images[:64, :, :, :],
                                                 tf.expand_dims(self.masks[:64, :, :], 3),
                                                 self.bg_images[:64, :, :, :],
                                                 8, "train_FG_on_text")
        img_sum4 = self.visualize_one_superimage(fg_x[64:128, :, :, :],
                                                 self.images[64:128, :, :, :],
                                                 tf.expand_dims(self.masks[64:128, :, :], 3),
                                                 self.bg_images[64:128, :, :, :],
                                                 8, "test_FG_on_text")

        # c randomly samples from Gaussian
        noise_c = self.model.con_latent_dist.sample_prior(self.batch_size)
        noise_x = self.model.get_generator(tf.concat(1, [noise_c, z]))
        img_sum5 = self.visualize_one_superimage(noise_x[:64, :, :, :],
                                                 self.images[:64, :, :, :],
                                                 tf.expand_dims(self.masks[:64, :, :], 3),
                                                 self.bg_images[:64, :, :, :],
                                                 8, "image_on_noise")

        # c with all 0s
        bg_c = tf.zeros_like(noise_c)
        bg_x = self.model.get_generator(tf.concat(1, [bg_c, z]))
        img_sum6 = self.visualize_one_superimage(bg_x[:64, :, :, :],
                                                 self.images[:64, :, :, :],
                                                 tf.expand_dims(self.masks[:64, :, :], 3),
                                                 self.bg_images[:64, :, :, :],
                                                 8, "image_for_BG")
        self.image_summary = tf.merge_summary([img_sum1, img_sum2, img_sum3, img_sum4, img_sum5, img_sum6])

    def preprocess(self, embeddings):
        # make sure every row with 10 column have the same embeddings
        for i in range(8):
            for j in range(1, 8):
                embeddings[i * 8 + j] = embeddings[i * 8]
                # images[i * 8 + j] = images[i * 8]
        return embeddings

    def go_save(self, samples, metadata, path, dataset, num_copy):
        row_imgs = {}
        for r in range(metadata.caption_num):
            row_imgs[metadata.filenames[r]] = []

        for r in range(metadata.caption_num):
            key = metadata.filenames[r]
            row_img = [metadata.images[r, :, :, :]]
            for c in range(num_copy):
                row_img.append(samples[r * num_copy + c, :, :, :])
            row_img = np.concatenate(row_img, axis=1)
            row_imgs[key].append(row_img)
        for key in row_imgs:
            img = np.concatenate(row_imgs[key], axis=0)
            # print('%s/%s.jpg' % (path, key), img.shape)
            # img_name = metadata.attributes[???]
            scipy.misc.imsave('%s/%s_%s.jpg' % (path, key, dataset), img)

    def epoch_save_samples(self, sess, num_copy):
        embeddings_train = np.repeat(self.dataset.fixedvisual_train.embeddings, num_copy, axis=0)
        embeddings_test = np.repeat(self.dataset.fixedvisual_test.embeddings, num_copy, axis=0)
        embeddings = np.concatenate([embeddings_train, embeddings_test], axis=0)
        if embeddings.shape[0] < self.batch_size:
            _, _, embeddings_pad, _, _ = self.dataset.test.next_batch(self.batch_size - embeddings.shape[0])
            embeddings = np.concatenate([embeddings, embeddings_pad], axis=0)
        feed_dict = {self.embeddings: embeddings}
        gen_samples = sess.run(self.fake_x, feed_dict)

        # save train
        num_train = self.dataset.fixedvisual_train.caption_num * num_copy
        self.go_save(gen_samples[:num_train], self.dataset.fixedvisual_train,
                     self.log_dir, 'train', num_copy)
        # save test
        num_test = self.dataset.fixedvisual_test.caption_num * num_copy
        self.go_save(gen_samples[num_train:num_train + num_test],
                     self.dataset.fixedvisual_test,
                     self.log_dir, 'test', num_copy)

    def epoch_sum_images(self, sess):
        images_train, masks_train, embeddings_train, bg_images_train, _ = self.dataset.train.next_batch(64)
        embeddings_train = self.preprocess(embeddings_train)

        images_test, masks_test, embeddings_test, bg_images_test, _ = self.dataset.test.next_batch(64)
        embeddings_test = self.preprocess(embeddings_test)

        images = np.concatenate([images_train, images_test], axis=0)
        masks = np.concatenate([masks_train, masks_test], axis=0)
        embeddings = np.concatenate([embeddings_train, embeddings_test], axis=0)
        bg_images = np.concatenate([bg_images_train, bg_images_test], axis=0)
        if self.batch_size > 128:
            images_pad, masks_pad, embeddings_pad, bg_images_pad, _ = self.dataset.test.next_batch(self.batch_size - 128)
            images = np.concatenate([images, images_pad], axis=0)
            masks = np.concatenate([masks, masks_pad], axis=0)
            embeddings = np.concatenate([embeddings, embeddings_pad], axis=0)
            bg_images = np.concatenate([bg_images, bg_images_pad], axis=0)
        feed_dict = {self.images: images,
                     self.masks: masks.astype(np.float32),
                     self.embeddings: embeddings,
                     self.bg_images: bg_images
                     }
        return sess.run(self.image_summary, feed_dict)

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

                log_keys = ["d_loss", "bg_d_loss", "fg_d_loss", "g_loss",
                            "e_loss", "e_kl_loss", "e_like_loss"]
                log_vars = []
                for k, v in self.log_vars:
                    if k in log_keys:
                        log_vars.append(v)

                for epoch in range(int(counter / self.updates_per_epoch), self.max_epoch):
                    widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(maxval=self.updates_per_epoch, widgets=widgets)
                    pbar.start()

                    all_log_vals = []
                    for i in range(self.updates_per_epoch):
                        pbar.update(i)
                        # training d
                        images, masks, embeddings, bg_images, _ = self.dataset.train.next_batch(self.batch_size)
                        # print(type(masks), masks.shape)
                        feed_dict = {self.images: images,
                                     self.masks: masks.astype(np.float32),
                                     self.embeddings: embeddings,
                                     self.bg_images: bg_images
                                     }
                        # ###TODO: Feed in once to save time
                        feed_out = [self.discriminator_trainer,
                                    self.bg_discriminator_trainer,
                                    self.fg_discriminator_trainer,
                                    # self.generator_trainer,
                                    # self.encoder_trainer,
                                    self.d_sum,
                                    self.bg_d_sum,
                                    self.fg_d_sum,
                                    self.hist_sum,
                                    log_vars,
                                    ]
                        _, _, _, d_summary, bg_d_summary, fg_d_summary,\
                            hist_summary, log_vals = sess.run(feed_out, feed_dict)
                        summary_writer.add_summary(d_summary, counter)
                        summary_writer.add_summary(bg_d_summary, counter)
                        summary_writer.add_summary(fg_d_summary, counter)
                        summary_writer.add_summary(hist_summary, counter)
                        # ###TODO: Feed in separately
                        # _, d_summary, log_vals, hist_summary = sess.run(
                        #     [self.discriminator_trainer, self.d_sum,
                        #      log_vars, self.hist_sum], feed_dict)
                        # summary_writer.add_summary(d_summary, counter)
                        # summary_writer.add_summary(hist_summary, counter)
                        # # training g&e&bg_d&fg_d
                        # _, bg_d_summary = sess.run(
                        #     [self.bg_discriminator_trainer, self.bg_d_sum], feed_dict
                        # )
                        # summary_writer.add_summary(bg_d_summary, counter)
                        # #
                        # _, fg_d_summary = sess.run(
                        #     [self.fg_discriminator_trainer, self.fg_d_sum], feed_dict
                        # )
                        # summary_writer.add_summary(fg_d_summary, counter)
                        # # ###*****************
                        #
                        _, g_summary = sess.run(
                            [self.generator_trainer, self.g_sum], feed_dict
                        )
                        summary_writer.add_summary(g_summary, counter)
                        #
                        _, e_summary = sess.run(
                            [self.encoder_trainer, self.e_sum], feed_dict
                        )
                        summary_writer.add_summary(e_summary, counter)

                        all_log_vals.append(log_vals)
                        counter += 1
                        if counter % self.snapshot_interval == 0:
                            snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                            fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                            print("Model saved in file: %s" % fn)

                    summary_writer.add_summary(self.epoch_sum_images(sess), counter)
                    self.epoch_save_samples(sess, cfg.TRAIN.NUM_COPY)

                    avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                    log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))
                    print("Epoch %d | " % (epoch) + log_line)
                    sys.stdout.flush()
                    if np.any(np.isnan(avg_log_vals)):
                        raise ValueError("NaN detected!")
