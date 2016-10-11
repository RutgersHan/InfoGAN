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
        c = self.model.generate_condition(embeddings)
        return c

    def get_interp_embeddings(self, embeddings):
        split0, split1 = tf.split(0, 2, embeddings)
        interp = tf.add(split0, split1) / 2.
        return tf.concat(0, [embeddings, interp])

    def init_opt(self):
        # self.images, self.masks, self.embeddings, self.bg_images
        self.build_placeholder()
        # masks is tf.float32 with 0s and 1s
        self.fg_images = tf.mul(self.images, tf.expand_dims(self.masks, 3))
        self.fg_wrong_images = tf.mul(self.wrong_images, tf.expand_dims(self.masks, 3))

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
            like_loss = self.compute_like_loss()
            generator_loss += like_loss

            # #######Total loss for build######################################
            self.log_vars.append(("g_loss", generator_loss))
            self.log_vars.append(("d_loss", discriminator_loss))
            self.prepare_trainer(generator_loss, discriminator_loss)
            # #######define self.g_sum, self.d_sum,....########################
            self.define_summaries()

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("model", reuse=True) as scope:
                self.visualization()
                print("success")

    # ####get discriminator_loss and generator_loss for FG#####################
    def compute_d_loss(self):
        real_d = self.model.get_discriminator(self.fg_images, self.embeddings)
        wrong_d = self.model.get_discriminator(self.fg_wrong_images, self.embeddings)
        fake_d = self.model.get_discriminator(self.fake_images, self.embeddings)

        real_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(real_d, tf.ones_like(real_d)))
        wrong_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(wrong_d, tf.zeros_like(wrong_d)))
        fake_d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fake_d, tf.zeros_like(fake_d)))

        discriminator_loss = real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
        self.log_vars.append(("d_loss_real", real_d_loss))
        self.log_vars.append(("d_loss_wrong", wrong_d_loss))
        self.log_vars.append(("d_loss_fake", fake_d_loss))
        return discriminator_loss

    def compute_like_loss(self):
        like_loss = tf.constant(0.)

        fake_L2_ftr = self.model.d_L2_template.construct(input=self.fake_images)
        real_L2_ftr = self.model.d_L2_template.construct(input=self.fg_images)
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

    def compute_g_loss(self):
        interp_fake_g = self.model.get_discriminator(self.interp_fake_images, self.interp_embeddings)

        generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(interp_fake_g, tf.ones_like(interp_fake_g)))
        self.log_vars.append(("g_loss_interp_fake", generator_loss))
        return generator_loss

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

    def visualization(self):
        c = self.sample_encoded_context(self.embeddings)
        z = tf.random_normal([self.batch_size, cfg.Z_DIM])
        fake_x = self.model.get_generator(tf.concat(1, [c, z]))
        fake_sum_train, superimage_train = self.visualize_one_superimage(fake_x[:64, :, :, :],
                                                                         self.fg_images[:64, :, :, :],
                                                                         8, "train_on_text")
        fake_sum_test, superimage_test = self.visualize_one_superimage(fake_x[64:128, :, :, :],
                                                                       self.fg_images[64:128, :, :, :],
                                                                       8, "test_on_text")
        self.superimages = tf.concat(0, [superimage_train, superimage_test])
        self.image_summary = tf.merge_summary([fake_sum_train, fake_sum_test])

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
                        generator_learning_rate *= 0.2
                        discriminator_learning_rate *= 0.2

                    all_log_vals = []
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
                        feed_out = [self.discriminator_trainer,
                                    self.d_sum,
                                    self.hist_sum,
                                    log_vars,
                                    ]
                        _, d_summary, hist_summary, log_vals = sess.run(feed_out, feed_dict)
                        summary_writer.add_summary(d_summary, counter)
                        summary_writer.add_summary(hist_summary, counter)
                        all_log_vals.append(log_vals)
                        # train g
                        feed_out = [self.generator_trainer,
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
