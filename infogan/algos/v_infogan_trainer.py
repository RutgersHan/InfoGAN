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

        self.fg_discriminator_trainer = None
        self.fg_generator_trainer = None
        self.images = None
        self.masks = None
        self.embeddings = None
        # self.bg_images = None
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
        # self.bg_images = tf.placeholder(
        #    tf.float32, [self.batch_size] + self.dataset.image_shape,
        #    name='real_bg_images')

    def sample_encoded_context(self, bencode=True):
        '''Helper function for init_opt'''
        if bencode:
            c_mean_logsigma = self.model.generate_fg_condition(self.embeddings)
            self.log_vars.append(("fg_hist_c_mean", c_mean_logsigma[0]))
            self.log_vars.append(("fg_hist_c_log_sigma", c_mean_logsigma[1]))
            # c_sample = self.model.con_latent_dist.sample_prior(self.batch_size)
            # c = c_mean_logsigma[0] + c_mean_logsigma[1] * c_sample
            mean = c_mean_logsigma[0]
            epsilon = tf.random_normal(tf.shape(mean))
            stddev = tf.exp(c_mean_logsigma[1])
            c = mean + stddev * epsilon

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
            z = self.model.latent_dist.sample_prior(self.batch_size)
            self.log_vars.append(("hist_z", z))
            # fg_c sampled from encoded text for generating fg_fake_x
            fg_c, fg_kl_loss = self.sample_encoded_context(bencode=True)
            fg_encoder_loss = fg_kl_loss
            self.log_vars.append(("fg_hist_fake_c", fg_c))
            self.log_vars.append(("fg_e_kl_loss", fg_kl_loss))
            # fg_noise_c randomly samples from Gaussian for generating fg_noise_x
            fg_noise_c = self.model.con_latent_dist.sample_prior(self.batch_size)
            self.log_vars.append(("fg_hist_noise_c", fg_noise_c))

            # ####get output from G network####################################
            self.fg_fake_x = self.model.get_fg_generator(tf.concat(1, [fg_c, z]))
            self.fg_noise_x = self.model.get_fg_generator(tf.concat(1, [fg_noise_c, z]))
            # self.fg_fake_x = self.model.get_fg_generator(fg_c)
            # self.fg_noise_x = self.model.get_fg_generator(fg_noise_c)

            # ####get discriminator_loss and generator_loss for FG##############
            # ####Different discriminators but the same generator###############
            fg_discriminator_loss, fg_generator_loss = self.compute_loss_from_fg_D()
            if cfg.TRAIN.MASK_FLAG and cfg.TRAIN.COEFF.LIKE > TINY:
                fg_like_loss = self.compute_like_loss_from_fg_D()
                fg_encoder_loss += cfg.TRAIN.COEFF.LIKE * fg_like_loss
                self.log_vars.append(("fg_e_like_loss_reweight", cfg.TRAIN.COEFF.LIKE * fg_like_loss))
                # ##Add like loss to ......
                # fg_generator_loss + cfg.TRAIN.COEFF.LIKE * fg_like_loss
                # fg_discriminator_loss += cfg.TRAIN.COEFF.LIKE * fg_like_loss  # TODO: whether add this ??

            # ##MI##############################################################
            if cfg.TRAIN.COEFF.REAL_C > TINY:
                mi_sum = tf.constant(0.)

                fg_real_c_var_dist_info = self.model.reconstuct_context(self.fg_images)
                fg_real_c_mi = self.computeMI(self.model.con_latent_dist, fg_c, fg_real_c_var_dist_info, 0)
                self.log_vars.append(("fg_d_g_real_c_mi", fg_real_c_mi))
                mi_sum += cfg.TRAIN.COEFF.REAL_C * fg_real_c_mi

                fg_fake_c_var_dist_info = self.model.reconstuct_context(self.fg_fake_x)
                fg_fake_c_mi = self.computeMI(self.model.con_latent_dist, fg_c, fg_fake_c_var_dist_info, 0)
                self.log_vars.append(("fg_d_g_fake_c_mi", fg_fake_c_mi))
                mi_sum += cfg.TRAIN.COEFF.REAL_C * fg_fake_c_mi

                self.log_vars.append(("fg_d_g_c_MI_weighted_sum", -mi_sum))
                fg_discriminator_loss -= mi_sum
                fg_generator_loss -= mi_sum

            # #######Total loss for build#######################################
            self.log_vars.append(("fg_e_loss", fg_encoder_loss))
            self.log_vars.append(("fg_g_loss", fg_generator_loss))
            self.log_vars.append(("fg_d_loss", fg_discriminator_loss))
            self.prepare_trainer(fg_encoder_loss, fg_generator_loss, fg_discriminator_loss)
            # #######define self.e_fg_sum, self.g_fg_sum, self.d_fg_sum,################
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
    def compute_like_loss_from_fg_D(self):
        # ##like loss based on element-wise distance between real and fake images
        # like_loss = tf.reduce_mean(tf.square(self.fg_images - self.fg_fake_x)) / 2.

        # ##like loss based on feature distance between real and fake images
        fg_real_shared_layers = self.model.get_fg_discriminator_shared(self.fg_images)
        fg_fake_shared_layers = self.model.get_fg_discriminator_shared(self.fg_fake_x)
        like_loss = tf.reduce_mean(tf.square(fg_real_shared_layers - fg_fake_shared_layers)) / 2.
        self.log_vars.append(("fg_e_like_loss", like_loss))
        return like_loss

    # ####get discriminator_loss and generator_loss for FG#####################
    def compute_loss_from_fg_D(self):
        fg_real_shared_layers = self.model.get_fg_discriminator_shared(self.fg_images)
        fg_fake_shared_layers = self.model.get_fg_discriminator_shared(self.fg_fake_x)
        # fg_noise_shared_layers = self.model.get_fg_discriminator_shared(self.fg_noise_x)

        fg_real_d = self.model.get_fg_discriminator(fg_real_shared_layers)
        fg_fake_d = self.model.get_fg_discriminator(fg_fake_shared_layers)
        # fg_noise_d = self.model.get_fg_discriminator(fg_noise_shared_layers)

        # Only real FG images are real images, all others are fake images
        fg_d_loss_legit = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fg_real_d, tf.ones_like(fg_real_d)))
        fg_d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fg_fake_d, tf.zeros_like(fg_fake_d)))
        # fg_d_loss_noise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fg_noise_d, tf.zeros_like(fg_noise_d)))

        # fg_discriminator_loss = fg_d_loss_legit + (fg_d_loss_fake + fg_d_loss_noise) / 2.
        fg_discriminator_loss = fg_d_loss_legit + fg_d_loss_fake
        self.log_vars.append(("fg_d_loss_real", fg_d_loss_legit))
        self.log_vars.append(("fg_d_loss_fake", fg_d_loss_fake))
        # self.log_vars.append(("fg_d_loss_noise", fg_d_loss_noise))

        fg_g_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fg_fake_d, tf.ones_like(fg_fake_d)))
        # fg_g_loss_noise = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fg_noise_d, tf.ones_like(fg_noise_d)))
        # fg_generator_loss = fg_g_loss_fake + fg_g_loss_noise
        fg_generator_loss = fg_g_loss_fake
        self.log_vars.append(("fg_g_loss_fake", fg_g_loss_fake))
        # self.log_vars.append(("fg_g_loss_noise", fg_g_loss_noise))
        return fg_discriminator_loss, fg_generator_loss

    def prepare_trainer(self, fg_encoder_loss, fg_generator_loss, fg_discriminator_loss):
        '''Helper function for init_opt'''
        all_vars = tf.trainable_variables()

        e_vars = [var for var in all_vars if
                  var.name.startswith('fg_e_')]
        g_vars = [var for var in all_vars if
                  var.name.startswith('fg_g_')]
        d_vars = [var for var in all_vars if
                  var.name.startswith('fg_d_')]
        r_vars = [var for var in all_vars if
                  var.name.startswith('fg_r_')]

        generator_optimizer = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=0.5)
        self.fg_generator_trainer = pt.apply_optimizer(generator_optimizer,
                                                       losses=[fg_encoder_loss + fg_generator_loss],
                                                       var_list=e_vars + g_vars)
        discriminator_optimizer = tf.train.AdamOptimizer(self.discriminator_learning_rate, beta1=0.5)
        self.fg_discriminator_trainer = pt.apply_optimizer(discriminator_optimizer,
                                                           losses=[fg_discriminator_loss],
                                                           var_list=d_vars + r_vars)

    def define_summaries(self):
        '''Helper function for init_opt'''
        all_sum = {'fg_e': [], 'fg_g': [], 'fg_d': [], 'fg_hist': []}
        for k, v in self.log_vars:
            if k.startswith('fg_e'):
                all_sum['fg_e'].append(tf.scalar_summary(k, v))
            elif k.startswith('fg_g'):
                all_sum['fg_g'].append(tf.scalar_summary(k, v))
            elif k.startswith('fg_d'):
                all_sum['fg_d'].append(tf.scalar_summary(k, v))
            elif k.startswith('fg_hist'):
                all_sum['fg_hist'].append(tf.histogram_summary(k, v))

        self.fg_e_sum = tf.merge_summary(all_sum['fg_e'])
        self.fg_g_sum = tf.merge_summary(all_sum['fg_g'])
        self.fg_d_sum = tf.merge_summary(all_sum['fg_d'])
        self.fg_hist_sum = tf.merge_summary(all_sum['fg_hist'])

    def visualize_one_superimage(self, img_var, images, fg_images, rows, filename):
        stacked_img = []
        stacked_real_fg = []
        for row in range(rows):
            img = images[row * rows, :, :, :]
            row_img = [img]  # real image and masked images
            row_real_fg = [img]
            for col in range(rows):
                row_img.append(img_var[row * rows + col, :, :, :])
                row_real_fg.append(fg_images[row * rows + col, :, :, :])
            # each rows is 1realimage +10_fakeimage
            stacked_img.append(tf.concat(1, row_img))
            stacked_real_fg.append(tf.concat(1, row_real_fg))
        imgs = tf.expand_dims(tf.concat(0, stacked_img), 0)
        real_fgs = tf.expand_dims(tf.concat(0, stacked_real_fg), 0)
        current_img_summary = tf.image_summary(filename, tf.concat(0, [imgs, real_fgs]))
        return current_img_summary

    def visualization(self):
        z = self.model.latent_dist.sample_prior(self.batch_size)
        # c sampled from encoded text
        fg_c, _ = self.sample_encoded_context(bencode=True)
        # fg_fake_x = self.model.get_fg_generator(fg_c)
        fg_fake_x = self.model.get_fg_generator(tf.concat(1, [fg_c, z]))
        fg_fake_sum1 = self.visualize_one_superimage(fg_fake_x[:64, :, :, :],
                                                     self.images[:64, :, :, :],
                                                     self.fg_images[:64, :, :, :],
                                                     8, "train_FG_on_text")
        fg_fake_sum2 = self.visualize_one_superimage(fg_fake_x[64:128, :, :, :],
                                                     self.images[64:128, :, :, :],
                                                     self.fg_images[64:128, :, :, :],
                                                     8, "test_FG_on_text")

        # c randomly samples from Gaussian
        fg_noise_c = self.model.con_latent_dist.sample_prior(self.batch_size)
        # fg_noise_x = self.model.get_fg_generator(fg_noise_c)
        fg_noise_x = self.model.get_fg_generator(tf.concat(1, [fg_noise_c, z]))
        fg_noise_sum = self.visualize_one_superimage(fg_noise_x[:64, :, :, :],
                                                     self.images[:64, :, :, :],
                                                     self.fg_images[:64, :, :, :],
                                                     8, "FG_on_noise")

        self.image_summary = tf.merge_summary([fg_fake_sum1, fg_fake_sum2, fg_noise_sum])

    def preprocess(self, x):
        # make sure every row with 10 column have the same embeddings
        for i in range(8):
            for j in range(1, 8):
                x[i * 8 + j] = x[i * 8]
        return x

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
            scipy.misc.imsave('%s/%s_%s_FG.jpg' % (path, key, dataset), img)

    def epoch_save_samples(self, sess, num_copy):
        # same embedding each row (num_copy), give same mean and log_sigma
        # but different sampled fg_c from sample_encoded_context()
        embeddings_train = np.repeat(self.dataset.fixedvisual_train.embeddings, num_copy, axis=0)
        embeddings_test = np.repeat(self.dataset.fixedvisual_test.embeddings, num_copy, axis=0)
        embeddings = np.concatenate([embeddings_train, embeddings_test], axis=0)
        if embeddings.shape[0] < self.batch_size:
            _, _, embeddings_pad, _, _ = self.dataset.test.next_batch(self.batch_size - embeddings.shape[0])
            embeddings = np.concatenate([embeddings, embeddings_pad], axis=0)
        feed_dict = {self.embeddings: embeddings}
        gen_samples = sess.run(self.fg_fake_x, feed_dict)

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
        images_train, masks_train, embeddings_train, _, _ = self.dataset.train.next_batch(64)
        images_train = self.preprocess(images_train)
        masks_train = self.preprocess(masks_train)
        embeddings_train = self.preprocess(embeddings_train)

        images_test, masks_test, embeddings_test, _, _ = self.dataset.test.next_batch(64)
        images_test = self.preprocess(images_test)
        masks_test = self.preprocess(masks_test)
        embeddings_test = self.preprocess(embeddings_test)

        images = np.concatenate([images_train, images_test], axis=0)
        masks = np.concatenate([masks_train, masks_test], axis=0)
        embeddings = np.concatenate([embeddings_train, embeddings_test], axis=0)

        if self.batch_size > 128:
            images_pad, masks_pad, embeddings_pad, _, _ = self.dataset.test.next_batch(self.batch_size - 128)
            images = np.concatenate([images, images_pad], axis=0)
            masks = np.concatenate([masks, masks_pad], axis=0)
            embeddings = np.concatenate([embeddings, embeddings_pad], axis=0)
        feed_dict = {self.images: images,
                     self.masks: masks.astype(np.float32),
                     self.embeddings: embeddings
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

                log_keys = ["fg_d_loss", "fg_g_loss", "fg_e_loss", "fg_e_kl_loss"]
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
                        images, masks, embeddings, _, _ = self.dataset.train.next_batch(self.batch_size)
                        feed_dict = {self.images: images,
                                     self.masks: masks.astype(np.float32),
                                     self.embeddings: embeddings
                                     }
                        if i % 1 == 0:
                            # ###TODO: Feed in once to save time
                            feed_out = [self.fg_discriminator_trainer,
                                        self.fg_d_sum,
                                        self.fg_hist_sum,
                                        log_vars,
                                        ]
                            _, fg_d_summary, fg_hist_summary, log_vals = sess.run(feed_out, feed_dict)
                            summary_writer.add_summary(fg_d_summary, counter)
                            summary_writer.add_summary(fg_hist_summary, counter)
                            all_log_vals.append(log_vals)
                        #
                        _, fg_g_summary, fg_e_summary = sess.run(
                            [self.fg_generator_trainer, self.fg_g_sum, self.fg_e_sum],
                            feed_dict
                        )
                        summary_writer.add_summary(fg_g_summary, counter)
                        summary_writer.add_summary(fg_e_summary, counter)

                        counter += 1
                        if counter % self.snapshot_interval == 0:
                            snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                            fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                            print("Model saved in file: %s" % fn)

                    summary_writer.add_summary(self.epoch_sum_images(sess), counter)
                    # self.epoch_save_samples(sess, cfg.TRAIN.NUM_COPY)

                    avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                    log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))
                    print("Epoch %d | " % (epoch) + log_line)
                    # print("Epoch %d" % epoch)
                    sys.stdout.flush()
                    if np.any(np.isnan(avg_log_vals)):
                        raise ValueError("NaN detected!")
