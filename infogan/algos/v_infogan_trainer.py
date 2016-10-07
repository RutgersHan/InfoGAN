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
        self.fg_images = None
        # self.embeddings = None
        # self.bg_images = None
        self.log_vars = []

    def build_placeholder(self):
        '''Helper function for init_opt'''
        self.images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='real_images')
        self.masks = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape[:2],
            name='real_masks')
        # self.embeddings = tf.placeholder(
        #    tf.float32, [self.batch_size] + self.dataset.embedding_shape,
        #    name='conditional_embeddings'
        # )
        # self.bg_images = tf.placeholder(
        #    tf.float32, [self.batch_size] + self.dataset.image_shape,
        #    name='real_bg_images')

    def init_opt(self):
        # self.images, self.masks, self.embeddings, self.bg_images
        self.build_placeholder()
        # masks is tf.float32 with 0s and 1s
        self.fg_images = tf.mul(self.images, tf.expand_dims(self.masks, 3))

        with pt.defaults_scope(phase=pt.Phase.train):
            # ####prepare input for G #########################################
            z = self.model.latent_dist.sample_prior(self.batch_size)
            reg_z = self.model.reg_z(z)
            self.log_vars.append(("hist_reg_z", reg_z))

            # ####get output from G network####################################
            self.fake_x = self.model.get_generator(self.fg_images, z)

            # ####get discriminator_loss and generator_loss from D#############
            discriminator_loss, generator_loss = self.compute_loss_from_D()

            # #####Like loss###################################################
            if cfg.TRAIN.MASK_FLAG and cfg.TRAIN.COEFF.LIKE > TINY:
                like_loss = self.compute_like_loss_from_D()
                generator_loss += cfg.TRAIN.COEFF.LIKE * like_loss
                # discriminator_loss += cfg.TRAIN.COEFF.LIKE * like_loss  # TODO: whether add this ??
                self.log_vars.append(("g_like_loss_reweight", cfg.TRAIN.COEFF.LIKE * like_loss))

            # #######Total loss for build#######################################
            self.log_vars.append(("g_loss", generator_loss))
            self.log_vars.append(("d_loss", discriminator_loss))
            self.prepare_trainer(generator_loss, discriminator_loss)
            # #######define self.e_sum, self.g_sum, self.d_sum,################
            # #######self.bg_d_sum, self.fg_d_sum, self.hist_sum, self.other_sum##############
            self.define_summaries()

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("model", reuse=True) as scope:
                self.visualization()
                print("success")

    def computeMI(self, dist, reg_z, reg_dist_info, bprior):
        '''Helper function for init_opt'''
        log_q_c_given_x = dist.logli(reg_z, reg_dist_info)
        cross_ent = tf.reduce_mean(-log_q_c_given_x)
        if bprior:
            log_q_c = dist.logli_prior(reg_z)
            ent = tf.reduce_mean(-log_q_c)
            mi_est = ent - cross_ent
        else:
            mi_est = -cross_ent
        # normalize by dimention
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
        self.log_vars.append(("g_like_loss", like_loss))
        return like_loss

    # ####get discriminator_loss and generator_loss####################
    # from D (real image detector)#####################################
    def compute_loss_from_D(self):
        real_shared_layers = self.model.get_discriminator_shared(self.images)
        fake_shared_layers = self.model.get_discriminator_shared(self.fake_x)

        real_d = self.model.get_discriminator(real_shared_layers)
        fake_d = self.model.get_discriminator(fake_shared_layers)

        # Only images with both BG and FG are real images, all others are fake images
        d_loss_legit = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_d, tf.ones_like(real_d)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_d, tf.zeros_like(fake_d)))

        discriminator_loss = d_loss_legit + d_loss_fake
        self.log_vars.append(("d_loss_real", d_loss_legit))
        self.log_vars.append(("d_loss_fake", d_loss_fake))
        real_p = tf.nn.sigmoid(real_d)
        feak_p = tf.nn.sigmoid(fake_d)
        self.log_vars.append(("max_real_p", tf.reduce_max(real_p)))
        self.log_vars.append(("min_real_p", tf.reduce_min(real_p)))
        self.log_vars.append(("max_fake_p", tf.reduce_max(feak_p)))
        self.log_vars.append(("min_fake_p", tf.reduce_min(feak_p)))

        # Train genarator to generate images as real as possible
        generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_d, tf.ones_like(fake_d)))
        return discriminator_loss, generator_loss

    def prepare_trainer(self, generator_loss, discriminator_loss):
        '''Helper function for init_opt'''
        all_vars = tf.trainable_variables()

        g_vars = [var for var in all_vars if
                  var.name.startswith('g_')]
        d_vars = [var for var in all_vars if
                  var.name.startswith('d_')]

        generator_optimizer = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=0.5)
        self.generator_trainer = pt.apply_optimizer(generator_optimizer,
                                                    losses=[generator_loss],
                                                    var_list=g_vars)
        discriminator_optimizer = tf.train.AdamOptimizer(self.discriminator_learning_rate, beta1=0.5)
        self.discriminator_trainer = pt.apply_optimizer(discriminator_optimizer,
                                                        losses=[discriminator_loss],
                                                        var_list=d_vars)

    def define_summaries(self):
        '''Helper function for init_opt'''
        all_sum = {'g': [], 'd': [], 'hist': []}
        for k, v in self.log_vars:
            if k.startswith('g_'):
                all_sum['g'].append(tf.scalar_summary(k, v))
            elif k.startswith('d_'):
                all_sum['d'].append(tf.scalar_summary(k, v))
            elif k.startswith('hist_'):
                all_sum['hist'].append(tf.histogram_summary(k, v))
            # else:
            #    all_sum['others'].append(tf.scalar_summary(k, v))

        self.g_sum = tf.merge_summary(all_sum['g'])
        self.d_sum = tf.merge_summary(all_sum['d'])
        self.hist_sum = tf.merge_summary(all_sum['hist'])
        # self.other_sum = tf.merge_summary(all_sum['others'])

    def visualize_one_superimage(self, img_var, images, fg_images, rows, filename):
        stacked_img = []
        for row in range(rows):
            img = images[row * rows, :, :, :]
            masked_img = fg_images[row * rows, :, :, :]
            row_img = [img, masked_img]  # real image and masked images
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

        fake_x = self.model.get_generator(self.fg_images, z)
        img_sum1 = self.visualize_one_superimage(fake_x[:64, :, :, :],
                                                 self.images[:64, :, :, :],
                                                 self.fg_images[:64, :, :, :],
                                                 8, "train_image_on_text")
        img_sum2 = self.visualize_one_superimage(fake_x[64:128, :, :, :],
                                                 self.images[64:128, :, :, :],
                                                 self.fg_images[64:128, :, :, :],
                                                 8, "test_image_on_text")

        self.image_summary = tf.merge_summary([img_sum1, img_sum2])

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
        images_train, masks_train, _, _, _ = self.dataset.train.next_batch(64)
        images_test, masks_test, _, _, _ = self.dataset.test.next_batch(64)

        images = np.concatenate([images_train, images_test], axis=0)
        masks = np.concatenate([masks_train, masks_test], axis=0)
        images = self.preprocess(images)
        masks = self.preprocess(masks)

        if self.batch_size > 128:
            images_pad, masks_pad, _, _, _ = self.dataset.test.next_batch(self.batch_size - 128)
            images = np.concatenate([images, images_pad], axis=0)
            masks = np.concatenate([masks, masks_pad], axis=0)
        feed_dict = {self.images: images,
                     self.masks: masks.astype(np.float32),
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
                else:
                    print("Created model with fresh parameters.")
                    sess.run(tf.initialize_all_variables())
                    counter = 0

                # summary_op = tf.merge_all_summaries()
                summary_writer = tf.train.SummaryWriter(self.log_dir, sess.graph)

                log_keys = ["d_loss", "g_loss"]
                log_vars = []
                for k, v in self.log_vars:
                    if k in log_keys:
                        log_vars.append(v)
                        print(k, v)

                for epoch in range(int(counter / self.updates_per_epoch), self.max_epoch):
                    widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(maxval=self.updates_per_epoch, widgets=widgets)
                    pbar.start()

                    all_log_vals = []
                    for i in range(self.updates_per_epoch):
                        pbar.update(i)
                        images, masks, _, _, _ = self.dataset.train.next_batch(self.batch_size)
                        # print(images.shape, masks.shape)
                        feed_dict = {self.images: images,
                                     self.masks: masks.astype(np.float32)
                                     }
                        if i % 6 == 0:
                            # training d
                            feed_out = [self.discriminator_trainer,
                                        self.d_sum,
                                        self.hist_sum,
                                        log_vars,
                                        ]
                            _, d_summary, hist_summary, log_vals = sess.run(feed_out, feed_dict)
                            summary_writer.add_summary(d_summary, counter)
                            summary_writer.add_summary(hist_summary, counter)
                            all_log_vals.append(log_vals)
                        #
                        _, g_summary = sess.run(
                            [self.generator_trainer, self.g_sum], feed_dict
                        )
                        summary_writer.add_summary(g_summary, counter)

                        counter += 1
                        if counter % self.snapshot_interval == 0:
                            snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                            fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                            print("Model saved in file: %s" % fn)

                    summary_writer.add_summary(self.epoch_sum_images(sess), counter)

                    avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                    log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))
                    print("Epoch %d | " % (epoch) + log_line)
                    sys.stdout.flush()
                    if np.any(np.isnan(avg_log_vals)):
                        raise ValueError("NaN detected!")
