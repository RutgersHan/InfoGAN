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
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.model_path = cfg.TRAIN.PRETRAINED_MODEL

        self.log_vars = []
        self.K = 2
        self.hr_image_shape = [self.dataset.image_shape[0] * self.K,
                               self.dataset.image_shape[1] * self.K,
                               self.dataset.image_shape[2]]
        self.hr_generator_learning_rate = 0.0002
        self.hr_discriminator_learning_rate = 0.0002

    def build_placeholder(self):
        '''Helper function for init_opt'''
        self.images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='real_images')
        self.wrong_images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='wrong_images'
        )
        self.hr_images = tf.placeholder(
            tf.float32, [self.batch_size] + self.hr_image_shape,
            name='real_hr_images')
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
        c = self.model.generate_condition(embeddings)
        return c

    def get_interp_embeddings(self, embeddings):
        split0, split1 = tf.split(0, 2, embeddings)
        interp = tf.add(split0, split1) / 2.
        return tf.concat(0, [embeddings, interp])

    def init_opt(self):
        # self.images, self.masks, self.embeddings, self.bg_images
        self.build_placeholder()
        lr_images = tf.image.resize_area(self.hr_images, self.dataset.image_shape[:2])
        self.back_hr_images = tf.image.resize_area(lr_images, self.hr_image_shape[:2])
        # self.back_images = tf.reshape(self.back_images, [self.batch_size] + self.hr_image_shape)

        with pt.defaults_scope(phase=pt.Phase.train):
            # ####get output from G network####################################
            c = self.sample_encoded_context(self.embeddings)
            z = tf.random_normal([self.batch_size, cfg.Z_DIM])
            self.log_vars.append(("hist_c", c))
            self.log_vars.append(("hist_z", z))
            self.fake_images = self.model.get_generator(tf.concat(1, [c, z]))

            self.interp_embeddings = self.get_interp_embeddings(self.embeddings)
            interp_c = self.sample_encoded_context(self.interp_embeddings)
            interp_z = tf.random_normal([int(self.batch_size * 3 / 2), cfg.Z_DIM])
            self.log_vars.append(("hist_interp_c", interp_c))
            self.log_vars.append(("hist_interp_z", interp_z))
            self.interp_fake_images = self.model.get_generator(tf.concat(1, [interp_c, interp_z]))

            # ####get discriminator_loss and generator_loss ###################
            discriminator_loss = self.compute_d_loss()
            generator_loss = self.compute_g_loss()
            self.log_vars.append(("g_loss", generator_loss))
            self.log_vars.append(("d_loss", discriminator_loss))

            # #### For hr_g and hr_d ##################################
            self.hr_fake_images = self.model.hr_get_generator(self.images)
            self.back_images = tf.image.resize_area(self.images, self.hr_image_shape[:2])

            self.hr_fake2_images = self.model.hr_get_generator(self.fake_images)
            self.back_fake_images = tf.image.resize_area(self.fake_images, self.hr_image_shape[:2])
            #
            hr_discriminator_loss = self.hr_compute_d_loss()
            hr_generator_loss = self.hr_compute_g_loss()
            self.log_vars.append(("hr_g_loss", hr_generator_loss))
            self.log_vars.append(("hr_d_loss", hr_discriminator_loss))

            # #######define self.g_sum, self.d_sum,....########################
            self.prepare_trainer(generator_loss, discriminator_loss,
                                 hr_generator_loss, hr_discriminator_loss)
            self.define_summaries()

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("model", reuse=True) as scope:
                self.visualization(cfg.TRAIN.NUM_COPY)
                print("success")

    # ####get discriminator_loss and generator_loss for FG#####################
    def compute_d_loss(self):
        real_d = self.model.get_discriminator(self.images, self.embeddings)
        wrong_d = self.model.get_discriminator(self.wrong_images, self.embeddings)
        fake_d = self.model.get_discriminator(self.fake_images, self.embeddings)

        real_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_d, tf.ones_like(real_d)))
        wrong_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(wrong_d, tf.zeros_like(wrong_d)))
        fake_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_d, tf.zeros_like(fake_d)))

        discriminator_loss = real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
        self.log_vars.append(("d_loss_real", real_d_loss))
        self.log_vars.append(("d_loss_wrong", wrong_d_loss))
        self.log_vars.append(("d_loss_fake", fake_d_loss))
        return discriminator_loss

    def compute_g_loss(self):
        interp_fake_g = self.model.get_discriminator(self.interp_fake_images, self.interp_embeddings)

        generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(interp_fake_g, tf.ones_like(interp_fake_g)))
        self.log_vars.append(("g_loss_interp_fake", generator_loss))
        return generator_loss

    def hr_compute_d_loss(self):
        real_d = self.model.hr_get_discriminator(self.hr_images, self.back_hr_images)
        fake_d = self.model.hr_get_discriminator(self.hr_fake_images, self.back_images)
        fake2_d = self.model.hr_get_discriminator(self.hr_fake2_images, self.back_fake_images)

        real_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_d, tf.ones_like(real_d)))
        fake_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_d, tf.zeros_like(fake_d)))
        fake2_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake2_d, tf.zeros_like(fake2_d)))
        #
        self.log_vars.append(("hr_d_loss_real", real_d_loss))
        self.log_vars.append(("hr_d_loss_fake", fake_d_loss))
        self.log_vars.append(("hr_d_loss_fake2", fake2_d_loss))
        discriminator_loss = real_d_loss + (fake_d_loss + fake2_d_loss) / 2.

        return discriminator_loss

    def hr_compute_g_loss(self):
        fake_g = self.model.hr_get_discriminator(self.hr_fake_images, self.back_images)
        fake2_g = self.model.hr_get_discriminator(self.hr_fake2_images, self.back_fake_images)
        fake_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_g, tf.ones_like(fake_g)))
        fake2_g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake2_g, tf.ones_like(fake2_g)))
        self.log_vars.append(("hr_g_loss_fake", fake_g_loss))
        self.log_vars.append(("hr_g_loss_fake2", fake2_g_loss))
        generator_loss = fake_g_loss + fake2_g_loss

        return generator_loss

    def prepare_trainer(self, generator_loss, discriminator_loss,
                        hr_generator_loss, hr_discriminator_loss):
        '''Helper function for init_opt'''
        all_vars = tf.trainable_variables()

        g_vars = [var for var in all_vars if
                  var.name.startswith('g_')]
        d_vars = [var for var in all_vars if
                  var.name.startswith('d_')]

        hr_g_vars = [var for var in all_vars if
                     var.name.startswith('hr_g_')]
        hr_d_vars = [var for var in all_vars if
                     var.name.startswith('hr_d_')]

        generator_optimizer = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=0.5)
        self.generator_trainer = pt.apply_optimizer(generator_optimizer,
                                                    losses=[generator_loss],
                                                    var_list=g_vars)
        discriminator_optimizer = tf.train.AdamOptimizer(self.discriminator_learning_rate, beta1=0.5)
        self.discriminator_trainer = pt.apply_optimizer(discriminator_optimizer,
                                                        losses=[discriminator_loss],
                                                        var_list=d_vars)

        hr_generator_optimizer = tf.train.AdamOptimizer(self.hr_generator_learning_rate, beta1=0.5)
        self.hr_generator_trainer = pt.apply_optimizer(hr_generator_optimizer,
                                                       losses=[hr_generator_loss],
                                                       var_list=hr_g_vars)
        hr_discriminator_optimizer = tf.train.AdamOptimizer(self.hr_discriminator_learning_rate, beta1=0.5)
        self.hr_discriminator_trainer = pt.apply_optimizer(hr_discriminator_optimizer,
                                                           losses=[hr_discriminator_loss],
                                                           var_list=hr_d_vars)

        self.log_vars.append(("g_learning_rate", self.generator_learning_rate))
        self.log_vars.append(("d_learning_rate", self.discriminator_learning_rate))

    def define_summaries(self):
        '''Helper function for init_opt'''
        all_sum = {'g': [], 'd': [], 'hr_g': [], 'hr_d': [], 'hist': []}
        for k, v in self.log_vars:
            if k.startswith('g'):
                all_sum['g'].append(tf.scalar_summary(k, v))
            elif k.startswith('d'):
                all_sum['d'].append(tf.scalar_summary(k, v))
            elif k.startswith('hist'):
                all_sum['hist'].append(tf.histogram_summary(k, v))
            elif k.startswith('hr_g'):
                all_sum['hr_g'].append(tf.scalar_summary(k, v))
            elif k.startswith('hr_d'):
                all_sum['hr_d'].append(tf.scalar_summary(k, v))

        self.g_sum = tf.merge_summary(all_sum['g'])
        self.d_sum = tf.merge_summary(all_sum['d'])
        self.hist_sum = tf.merge_summary(all_sum['hist'])

        self.hr_g_sum = tf.merge_summary(all_sum['hr_g'])
        self.hr_d_sum = tf.merge_summary(all_sum['hr_d'])

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

    def visualization(self, n):
        c = self.sample_encoded_context(self.embeddings)
        z = tf.random_normal([self.batch_size, cfg.Z_DIM])
        fake_x = self.model.get_generator(tf.concat(1, [c, z]))
        fake_sum_train, superimage_train = self.visualize_one_superimage(fake_x[:n * n, :, :, :],
                                                                         self.images[:n * n, :, :, :],
                                                                         n, "train_on_text")
        fake_sum_test, superimage_test = self.visualize_one_superimage(fake_x[n * n:2 * n * n, :, :, :],
                                                                       self.images[n * n:2 * n * n, :, :, :],
                                                                       n, "test_on_text")
        self.superimages = tf.concat(0, [superimage_train, superimage_test])
        self.image_summary = tf.merge_summary([fake_sum_train, fake_sum_test])

        # From 64 * 64 to 128 * 128
        hr_fake_x = self.model.hr_get_generator(fake_x)

        hr_fake_sum_train, hr_superimage_train = self.visualize_one_superimage(hr_fake_x[:n * n, :, :, :],
                                                                               self.hr_images[:n * n, :, :, :],
                                                                               n, "hr_train_on_text")
        hr_fake_sum_test, hr_superimage_test = self.visualize_one_superimage(hr_fake_x[n * n:2 * n * n, :, :, :],
                                                                             self.hr_images[n * n:2 * n * n, :, :, :],
                                                                             n, "hr_test_on_text")
        self.hr_superimages = tf.concat(0, [hr_superimage_train, hr_superimage_test])
        self.hr_image_summary = tf.merge_summary([hr_fake_sum_train, hr_fake_sum_test])

    def preprocess(self, x, n):
        # make sure every row with 10 column have the same embeddings
        for i in range(n):
            for j in range(1, n):
                x[i * n + j] = x[i * n]
        return x

    def epoch_sum_images(self, sess, n, bHR):
        images_train, _, hr_images_train, embeddings_train, captions_train, _, _ = self.dataset.train.next_batch(n * n, 4)
        images_train = self.preprocess(images_train, n)
        hr_images_train = self.preprocess(hr_images_train, n)
        embeddings_train = self.preprocess(embeddings_train, n)

        images_test, _, hr_images_test, embeddings_test, captions_test, _, _ = self.dataset.test.next_batch(n * n, 1)
        images_test = self.preprocess(images_test, n)
        hr_images_test = self.preprocess(hr_images_test, n)
        embeddings_test = self.preprocess(embeddings_test, n)

        images = np.concatenate([images_train, images_test], axis=0)
        hr_images = np.concatenate([hr_images_train, hr_images_test], axis=0)
        embeddings = np.concatenate([embeddings_train, embeddings_test], axis=0)

        if self.batch_size > 2 * n * n:
            images_pad, _, hr_images_pad, embeddings_pad, _, _, _ = self.dataset.test.next_batch(self.batch_size - 2 * n * n, 1)
            images = np.concatenate([images, images_pad], axis=0)
            hr_images = np.concatenate([hr_images, hr_images_pad], axis=0)
            embeddings = np.concatenate([embeddings, embeddings_pad], axis=0)

        prefix = 'lr_'
        if bHR:
            prefix = 'hr_'
            feed_dict = {self.hr_images: hr_images,
                         self.embeddings: embeddings
                         }
            gen_samples, img_summary = sess.run([self.hr_superimages, self.hr_image_summary], feed_dict)
        else:
            feed_dict = {self.images: images,
                         self.embeddings: embeddings
                         }
            gen_samples, img_summary = sess.run([self.superimages, self.image_summary], feed_dict)

        # save images generated for train and test captions
        scipy.misc.imsave('%s/%strain.jpg' % (self.log_dir, prefix), gen_samples[0])
        # pfi_train = open('%s/%strain.txt' % (self.log_dir, prefix), "w")

        scipy.misc.imsave('%s/%stest.jpg' % (self.log_dir, prefix), gen_samples[1])
        pfi_test = open('%s/%stest.txt' % (self.log_dir, prefix), "w")
        for row in range(n):
            # pfi_train.write('\n***row %d***\n' % row)
            # pfi_train.write(captions_train[row * n])

            pfi_test.write('\n***row %d***\n' % row)
            pfi_test.write(captions_test[row * n])
        # pfi_train.close()
        pfi_test.close()

        return img_summary

    def train(self):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                self.init_opt()

                sess.run(tf.initialize_all_variables())

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
                    # sess.run(tf.initialize_all_variables())
                    counter = 0

                # summary_op = tf.merge_all_summaries()
                summary_writer = tf.train.SummaryWriter(self.log_dir, sess.graph)

                updates_per_epoch = int(self.dataset.train._num_examples / self.batch_size)
                generator_learning_rate = cfg.TRAIN.GENERATOR_LR
                discriminator_learning_rate = cfg.TRAIN.DISCRIMINATOR_LR
                for epoch in range(int(counter / updates_per_epoch), self.max_epoch):
                    widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(maxval=updates_per_epoch, widgets=widgets)
                    pbar.start()
                    if epoch % 600 == 0:
                        bHR = 0
                        keys = ["d_loss", "g_loss"]
                        log_vars = []
                        log_keys = []
                        for k, v in self.log_vars:
                            if k in keys:
                                log_vars.append(v)
                                log_keys.append(k)
                        feed_out_d = [self.discriminator_trainer,
                                      self.d_sum,
                                      self.hist_sum,
                                      log_vars,
                                      ]
                        feed_out_g = [self.generator_trainer,
                                      self.g_sum,
                                      ]
                    elif epoch % 300 == 0:
                        bHR = 1
                        keys = ["hr_d_loss", "hr_g_loss"]
                        log_vars = []
                        log_keys = []
                        for k, v in self.log_vars:
                            if k in keys:
                                log_vars.append(v)
                                log_keys.append(k)
                        feed_out_d = [self.hr_discriminator_trainer,
                                      self.hr_d_sum,
                                      self.hist_sum,
                                      log_vars,
                                      ]
                        feed_out_g = [self.hr_generator_trainer,
                                      self.hr_g_sum,
                                      ]

                    # if epoch % 200 == 0 and epoch != 0:  # for noBN
                    if bHR == 0 and epoch % 100 == 0 and epoch != 0:  # for base: with BN
                        generator_learning_rate *= 0.5
                        discriminator_learning_rate *= 0.5

                    all_log_vals = []
                    for i in range(updates_per_epoch):
                        pbar.update(i)
                        # training d
                        images, wrong_images, hr_images, embeddings, _, _, _ = self.dataset.train.next_batch(self.batch_size, 4)
                        feed_dict = {self.images: images,
                                     self.wrong_images: wrong_images,
                                     self.hr_images: hr_images,
                                     self.embeddings: embeddings,
                                     self.generator_learning_rate: generator_learning_rate,
                                     self.discriminator_learning_rate: discriminator_learning_rate
                                     }
                        # train d

                        _, d_summary, hist_summary, log_vals = sess.run(feed_out_d, feed_dict)
                        summary_writer.add_summary(d_summary, counter)
                        summary_writer.add_summary(hist_summary, counter)
                        all_log_vals.append(log_vals)
                        # train g
                        _, g_summary = sess.run(
                            feed_out_g, feed_dict
                        )
                        summary_writer.add_summary(g_summary, counter)
                        # save checkpoint
                        counter += 1
                        if counter % self.snapshot_interval == 0:
                            snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                            fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                            print("Model saved in file: %s" % fn)

                    summary_writer.add_summary(self.epoch_sum_images(sess, cfg.TRAIN.NUM_COPY, bHR), counter)

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
