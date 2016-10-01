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
import scipy.misc

TINY = 1e-8
RECON_VS_GAN = 1.0  # 5.0  # 1.0 # 1e-6


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
                 generator_learning_rate=2e-4,  # 2e-4,  #1e-3
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
        self.masks = None
        self.embeddings = None
        self.log_vars = []

    def init_opt(self):
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
        self.z_noise = tf.placeholder(
            tf.float32, [self.batch_size] + [self.model.ef_dim],
            name='sample_noise'
        )
        self.z_noise_c_var = tf.placeholder(
            tf.float32, [self.batch_size] + [self.model.ef_dim],
            name='sample_c_var'
        )

        with pt.defaults_scope(phase=pt.Phase.train):
            c_var = self.model.generate_condition(self.embeddings)
            self.log_vars.append(("condition_mean", c_var[0]))
            self.log_vars.append(("condition_log_sigma", c_var[0]))
            z_c_var = c_var[0] + c_var[1] * self.z_noise_c_var

            z_bg = self.model.latent_dist.sample_prior(self.batch_size)
            reg_z = self.model.reg_z(z_bg)

            # ####KL loss #####################################################
            kl_loss = KL_loss(c_var[0], c_var[1])
            encoder_loss = kl_loss
            self.log_vars.append(("e_kl_loss", kl_loss))

            # ####d_loss_legit & d_loss_fake ##################################
            fake_x = self.model.get_generator(tf.concat(1, [z_c_var, z_bg]))
            self.fake_x = fake_x
            noise_x = self.model.get_generator(tf.concat(1, [self.z_noise, z_bg]))

            real_shared_layers = self.model.get_discriminator_shared(self.images)
            fake_shared_layers = self.model.get_discriminator_shared(fake_x)
            noise_shared_layers = self.model.get_discriminator_shared(noise_x)

            real_d = self.model.get_discriminator(real_shared_layers)
            fake_d = self.model.get_discriminator(fake_shared_layers)
            noise_d = self.model.get_discriminator(noise_shared_layers)

            d_loss_legit = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_d, tf.ones_like(real_d)))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_d, tf.zeros_like(fake_d)))
            d_loss_noise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(noise_d, tf.zeros_like(noise_d)))
            d_loss_fake = (d_loss_fake + d_loss_noise) / 2.

            discriminator_loss = d_loss_legit + d_loss_fake
            self.log_vars.append(("d_loss_fake", d_loss_fake))
            self.log_vars.append(("d_loss_noise", d_loss_noise))
            self.log_vars.append(("d_loss_real", d_loss_legit))

            real_p = tf.nn.sigmoid(real_d)
            feak_p = tf.nn.sigmoid(fake_d)
            self.log_vars.append(("max_real_p", tf.reduce_max(real_p)))
            self.log_vars.append(("min_real_p", tf.reduce_min(real_p)))
            self.log_vars.append(("max_fake_p", tf.reduce_max(feak_p)))
            self.log_vars.append(("min_fake_p", tf.reduce_min(feak_p)))

            # ####g_loss_fake & g_loss_noise ##################################
            g_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_d, tf.ones_like(fake_d)))
            g_loss_noise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(noise_d, tf.ones_like(noise_d)))
            generator_loss = (g_loss_fake + g_loss_noise) / 2.
            self.log_vars.append(("g_loss_fake", g_loss_fake))
            self.log_vars.append(("g_loss_noise", g_loss_noise))

            # #####Like loss###################################################
            # masks is tf.float32 with 0s and 1s
            real_masked_x = tf.mul(self.images, tf.expand_dims(self.masks, 3))
            fake_masked_x = tf.mul(fake_x, tf.expand_dims(self.masks, 3))
            # ##like loss based on element-wise distance between real and fake images
            # like_loss = tf.reduce_mean(tf.square(real_masked_x - fake_masked_x)) / 2.

            # ##like loss based on feature distance between real and fake images
            real_masked_shared_layers = self.model.get_discriminator_shared(real_masked_x)
            fake_masked_shared_layers = self.model.get_discriminator_shared(fake_masked_x)
            real_f = self.model.extract_features(real_masked_shared_layers)
            fake_f = self.model.extract_features(fake_masked_shared_layers)
            like_loss = tf.reduce_mean(tf.square(real_f - fake_f)) / 2.
            # ##Add like loss to ......
            encoder_loss += like_loss
            generator_loss += RECON_VS_GAN * like_loss
            self.log_vars.append(("e_like_loss", like_loss))
            self.log_vars.append(("g_like_loss_reweight", RECON_VS_GAN * like_loss))

            # #######MI between fake_bg and the prior z_bg#####################
            # bg images have strong shape information
            # fake_bg_x = tf.mul(fake_x, tf.expand_dims(1. - self.masks, 3))
            # fake_bg_shared_layers = self.model.get_discriminator_shared(fake_bg_x)
            #
            fake_bg_shared_layers = fake_shared_layers - fake_masked_shared_layers
            #
            fake_reg_z_dist_info = self.model.reconstuct_bg(fake_bg_shared_layers)
            # compute for discrete and continuous codes separately
            mi_est = tf.constant(0.)
            cross_ent = tf.constant(0.)
            # discrete:
            if len(self.model.reg_disc_latent_dist.dists) > 0:
                disc_reg_z = self.model.disc_reg_z(reg_z)
                disc_reg_dist_info = self.model.disc_reg_dist_info(fake_reg_z_dist_info)

                disc_log_q_c_given_x = self.model.reg_disc_latent_dist.logli(disc_reg_z, disc_reg_dist_info)
                disc_log_q_c = self.model.reg_disc_latent_dist.logli_prior(disc_reg_z)

                disc_cross_ent = tf.reduce_mean(-disc_log_q_c_given_x)
                disc_ent = tf.reduce_mean(-disc_log_q_c)
                disc_mi_est = disc_ent - disc_cross_ent

                mi_est += disc_mi_est
                cross_ent += disc_cross_ent
                self.log_vars.append(("MI_disc", disc_mi_est))
                self.log_vars.append(("CrossEnt_disc", disc_cross_ent))
                # ##Add MI to ......
                discriminator_loss -= self.info_reg_coeff * disc_mi_est
                generator_loss -= self.info_reg_coeff * disc_mi_est
                self.log_vars.append(("g_d_negMI_disc_reweight", -self.info_reg_coeff * disc_mi_est))
            # continuous
            if len(self.model.reg_cont_latent_dist.dists) > 0:
                cont_reg_z = self.model.cont_reg_z(reg_z)
                cont_reg_dist_info = self.model.cont_reg_dist_info(fake_reg_z_dist_info)

                cont_log_q_c_given_x = self.model.reg_cont_latent_dist.logli(cont_reg_z, cont_reg_dist_info)
                cont_log_q_c = self.model.reg_cont_latent_dist.logli_prior(cont_reg_z)

                cont_cross_ent = tf.reduce_mean(-cont_log_q_c_given_x)
                cont_ent = tf.reduce_mean(-cont_log_q_c)
                cont_mi_est = cont_ent - cont_cross_ent

                mi_est += cont_mi_est
                cross_ent += cont_cross_ent
                self.log_vars.append(("MI_cont", cont_mi_est))
                self.log_vars.append(("CrossEnt_cont", cont_cross_ent))
                # ##Add MI to ......
                discriminator_loss -= self.info_reg_coeff * cont_mi_est
                generator_loss -= self.info_reg_coeff * cont_mi_est
                self.log_vars.append(("g_d_negMI_cont_reweight", -self.info_reg_coeff * cont_mi_est))
            self.log_vars.append(("MI", mi_est))
            self.log_vars.append(("CrossEnt", cross_ent))

            # #######Total loss for each network###############################
            self.log_vars.append(("d_loss", discriminator_loss))
            self.log_vars.append(("g_loss", generator_loss))
            self.log_vars.append(("e_loss", encoder_loss))

            all_vars = tf.trainable_variables()

            e_vars = [var for var in all_vars if
                      var.name.startswith('c_')]
            d_vars = [var for var in all_vars if
                      var.name.startswith('d_')]
            g_vars = [var for var in all_vars if
                      var.name.startswith('g_')]
            r_vars = [var for var in all_vars if
                      var.name.startswith('r_')]

            discriminator_optimizer = tf.train.AdamOptimizer(self.discriminator_learning_rate, beta1=0.5)
            self.discriminator_trainer = pt.apply_optimizer(discriminator_optimizer,
                                                            losses=[discriminator_loss],
                                                            var_list=d_vars + r_vars)

            # Change by TX (2)
            generator_optimizer = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=0.5)
            # self.generator_trainer = pt.apply_optimizer(generator_optimizer, losses=[generator_loss + encoder_loss],
            #                                             var_list=g_vars + e_vars)
            self.generator_trainer = pt.apply_optimizer(generator_optimizer,
                                                        losses=[generator_loss],
                                                        var_list=g_vars)

            encoder_optimizer = tf.train.AdamOptimizer(self.encoder_learning_rate, beta1=0.5)
            self.encoder_trainer = pt.apply_optimizer(encoder_optimizer,
                                                      losses=[encoder_loss],
                                                      var_list=e_vars)
            # ****************************

            all_sum = {'g': [], 'd': [], 'e': [], 'condition': [], 'others': []}
            for k, v in self.log_vars:
                if k.startswith('condition'):
                    all_sum['condition'].append(tf.histogram_summary(k, v))
                elif k.startswith('g_'):
                    all_sum['g'].append(tf.scalar_summary(k, v))
                elif k.startswith('d_'):
                    all_sum['d'].append(tf.scalar_summary(k, v))
                elif k.startswith('e_'):
                    all_sum['e'].append(tf.scalar_summary(k, v))
                else:
                    all_sum['others'].append(tf.scalar_summary(k, v))

            self.g_sum = tf.merge_summary(all_sum['g'])
            self.d_sum = tf.merge_summary(all_sum['d'])
            self.e_sum = tf.merge_summary(all_sum['e'])
            self.other_sum = tf.merge_summary(all_sum['condition'] + all_sum['others'])

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("model", reuse=True) as scope:
                self.visualization()
                print("success")

    def generate_bg_pad(self, bg_img, bg_mask, dim_indices):
        nonzero_num = tf.reduce_sum(bg_mask, reduction_indices=dim_indices, keep_dims=True)
        value_sum = tf.reduce_sum(bg_img, reduction_indices=dim_indices, keep_dims=True)
        value_mean = tf.div(value_sum, nonzero_num)
        bg_pad = tf.mul(1. - bg_mask, tf.add(bg_img, value_mean))
        return bg_pad

    def extract_padded_bg_images(self, images, masks):
        bg_masks = 1.0 - masks
        bg_imgs = tf.mul(images, bg_masks)
        # bg_pad_mean = self.generate_bg_pad(bg_imgs, bg_masks, [1, 2])

        # bg_pad_rowmean = self.generate_bg_pad(bg_imgs, bg_masks, 1)
        # bg_pad_colmean = self.generate_bg_pad(bg_imgs, bg_masks, 2)
        # bg_imgs_padded = tf.add(bg_imgs, (bg_pad_rowmean + bg_pad_colmean) / 2.)

        # bg_imgs_padded = tf.add(bg_imgs, bg_pad_mean)
        return bg_imgs  # bg_imgs_padded

    def visualize_one_superimage(self, img_var, images, masks, rows, filename):
        bg_imgs_padded = self.extract_padded_bg_images(images, masks)
        fg_imgs = tf.mul(images, masks)

        stacked_img = []
        for row in range(rows):
            img = images[row * rows, :, :, :]
            masked_img = fg_imgs[row * rows, :, :, :]
            bg_img = bg_imgs_padded[row * rows, :, :, :]
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
        c_var = self.model.generate_condition(self.embeddings)
        z_c_var = c_var[0] + c_var[1] * self.z_noise_c_var

        z_bg_row = self.model.latent_dist.sample_prior(8)
        z_bg = tf.tile(z_bg_row, tf.constant([16, 1]))
        if self.batch_size > 128:
            z_bg_pad = self.model.latent_dist.sample_prior(self.batch_size - 128)
            z_bg = tf.concat(0, [z_bg, z_bg_pad])

        img_c_var = self.model.get_generator(tf.concat(1, [z_c_var, z_bg]))
        img_sum1 = self.visualize_one_superimage(img_c_var[:64, :, :, :],
                                                 self.images[:64, :, :, :],
                                                 tf.expand_dims(self.masks[:64, :, :], 3),
                                                 8, "train_image_on_text")
        img_sum2 = self.visualize_one_superimage(img_c_var[64:128, :, :, :],
                                                 self.images[64:128, :, :, :],
                                                 tf.expand_dims(self.masks[64:128, :, :], 3),
                                                 8, "test_image_on_text")

        img_noise = self.model.get_generator(tf.concat(1, [self.z_noise, z_bg]))
        img_sum3 = self.visualize_one_superimage(img_noise[:64, :, :, :],
                                                 self.images[:64, :, :, :],
                                                 tf.expand_dims(self.masks[:64, :, :], 3),
                                                 8, "image_on_noise")
        self.image_summary = tf.merge_summary([img_sum1, img_sum2, img_sum3])

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
        z_row = np.random.normal(0., 1., (num_copy - 1, self.model.ef_dim))
        z_row_mean = np.zeros((1, self.model.ef_dim), dtype=np.float32)
        z_row = np.concatenate([z_row_mean, z_row], axis=0)
        z1 = np.tile(z_row,
                     [embeddings.shape[0] / num_copy, 1])
        if embeddings.shape[0] < self.batch_size:
            z_pad = np.random.normal(0., 1., (self.batch_size - embeddings.shape[0], self.model.ef_dim))
            z1 = np.concatenate([z1, z_pad], axis=0)
            _, _, embeddings_pad, _ = self.dataset.test.next_batch(self.batch_size - embeddings.shape[0])
            embeddings = np.concatenate([embeddings, embeddings_pad], axis=0)

        feed_dict = {self.embeddings: embeddings,
                     self.z_noise_c_var: z1
                     }
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
        images_train, masks_train, embeddings_train, _ = self.dataset.train.next_batch(64)
        embeddings_train = self.preprocess(embeddings_train)

        images_test, masks_test, embeddings_test, _ = self.dataset.test.next_batch(64)
        embeddings_test = self.preprocess(embeddings_test)

        images = np.concatenate([images_train, images_test], axis=0)
        masks = np.concatenate([masks_train, masks_test], axis=0)
        embeddings = np.concatenate([embeddings_train, embeddings_test], axis=0)

        z1 = np.tile(np.random.normal(0., 1., (8, self.model.ef_dim)), [16, 1])
        z2 = np.random.normal(0., 1., (self.batch_size, self.model.ef_dim))
        if self.batch_size > 128:
            z_pad = np.random.normal(0., 1., (self.batch_size - 128, self.model.ef_dim))
            z1 = np.concatenate([z1, z_pad], axis=0)
            images_pad, masks_pad, embeddings_pad, _ = self.dataset.test.next_batch(self.batch_size - 128)
            images = np.concatenate([images, images_pad], axis=0)
            masks = np.concatenate([masks, masks_pad], axis=0)
            embeddings = np.concatenate([embeddings, embeddings_pad], axis=0)
        feed_dict = {self.images: images,
                     self.masks: masks.astype(np.float32),
                     self.embeddings: embeddings,
                     self.z_noise_c_var: z1,
                     self.z_noise: z2
                     }
        return sess.run(self.image_summary, feed_dict)

    def train(self):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            with tf.device("/gpu:0"):
                self.init_opt()
                # summary_op = tf.merge_all_summaries()
                summary_writer = tf.train.SummaryWriter(self.log_dir, sess.graph)

                saver = tf.train.Saver(tf.all_variables(), keep_checkpoint_every_n_hours=2)

                if self.model_path:
                    print("Reading model parameters from %s" % self.model_path)
                    saver.restore(sess, self.model_path)
                    counter = self.model_path[self.model_path.rfind('_') + 1:self.model_path.rfind('.')]
                    counter = int(counter)
                    self.epoch_save_samples(sess, 8)
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
                        images, masks, embeddings, _ = self.dataset.train.next_batch(self.batch_size)
                        # print(type(masks), masks.shape)
                        z1 = np.random.normal(0., 1., (self.batch_size, self.model.ef_dim))
                        z2 = np.random.normal(0., 1., (self.batch_size, self.model.ef_dim))
                        feed_dict_d = {self.images: images,
                                       self.masks: masks.astype(np.float32),
                                       self.embeddings: embeddings,
                                       self.z_noise_c_var: z1,
                                       self.z_noise: z2
                                       }
                        _, d_summary, log_vals, condition_vals, other_summary = sess.run(
                            [self.discriminator_trainer, self.d_sum,
                             log_vars, condition_var, self.other_sum], feed_dict_d)

                        summary_writer.add_summary(d_summary, counter)
                        summary_writer.add_summary(other_summary, counter)
                        # training g&e
                        images, masks, embeddings, _ = self.dataset.train.next_batch(self.batch_size)
                        z1 = np.random.normal(0., 1., (self.batch_size, self.model.ef_dim))
                        z2 = np.random.normal(0., 1., (self.batch_size, self.model.ef_dim))
                        feed_dict_ge = {self.images: images,
                                        self.masks: masks.astype(np.float32),
                                        self.embeddings: embeddings,
                                        self.z_noise_c_var: z1,
                                        self.z_noise: z2}
                        # Change by TX (2)
                        # _, g_summary, e_summary = sess.run(
                        #    [self.generator_trainer, self.g_sum, self.e_sum], feed_dict_ge
                        # )
                        _, g_summary = sess.run(
                            [self.generator_trainer, self.g_sum], feed_dict_ge
                        )
                        _, e_summary = sess.run(
                            [self.encoder_trainer, self.e_sum], feed_dict_ge
                        )
                        # *****************
                        summary_writer.add_summary(g_summary, counter)
                        summary_writer.add_summary(e_summary, counter)

                        all_log_vals.append(log_vals)
                        counter += 1

                        if counter % self.snapshot_interval == 0:
                            snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                            fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                            print("Model saved in file: %s" % fn)

                    summary_writer.add_summary(self.epoch_sum_images(sess), counter)
                    self.epoch_save_samples(sess, 8)

                    avg_log_vals = np.mean(np.array(all_log_vals), axis=0)

                    log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))
                    print("Epoch %d | " % (epoch) + log_line)
                    sys.stdout.flush()
                    if np.any(np.isnan(avg_log_vals)):
                        raise ValueError("NaN detected!")
