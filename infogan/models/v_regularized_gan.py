import prettytensor as pt
import tensorflow as tf
import infogan.misc.custom_ops
from infogan.misc.custom_ops import leaky_rectify
from infogan.misc.config import cfg

#  TODO: In discriminate
#  Does template.constrct really shared the computation, I did 3 times construct


class ConRegularizedGAN(object):
    def __init__(self, image_shape):
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.network_type = cfg.GAN.NETWORK_TYPE
        self.image_shape = image_shape
        self.gf_dim = cfg.GAN.GF_DIM
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM

        self.image_shape = image_shape
        self.s = image_shape[0]
        self.s2, self.s4, self.s8, self.s16 = int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)
        if cfg.GAN.NETWORK_TYPE == "default":
            with tf.variable_scope("g_net"):
                self.g_context_template = self.context_embedding()
                self.generator_template = self.generator()

            with tf.variable_scope("d_net"):
                self.d_start_template = self.start_discriminator()
                self.d_context_template = self.context_embedding()
                self.d_end_template = self.end_discriminator()
        else:
            raise NotImplementedError

    def context_embedding(self):
        template = (pt.template("input").
                    custom_fully_connected(self.ef_dim).
                    apply(leaky_rectify, leakiness=0.2))
        return template

    def generate_condition(self, c_var):
        conditions = self.g_context_template.construct(input=c_var)
        return conditions

    def generator(self):
        node1_0 = \
            (pt.template("input").
             custom_fully_connected(self.s16 * self.s16 * self.gf_dim * 8).
             fc_batch_norm().
             reshape([-1, self.s16, self.s16, self.gf_dim * 8]))
        node1_1 = \
            (node1_0.
             custom_conv2d(self.gf_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(tf.nn.relu))

        node2_0 = \
            (node1.
             custom_deconv2d([0, self.s8, self.s8, self.gf_dim * 4], k_h=4, k_w=4).
             conv_batch_norm())
        node2_1 = \
            (node2_0.
             custom_conv2d(self.gf_dim * 1, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 1, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node2 = \
            (node2_0.
             apply(tf.add, node2_1).
             apply(tf.nn.relu))

        template = \
            (node2.
             custom_deconv2d([0, self.s4, self.s4, self.gf_dim * 2], k_h=4, k_w=4).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s2, self.s2, self.gf_dim], k_h=4, k_w=4).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0] + list(self.image_shape), k_h=4, k_w=4).
             apply(tf.nn.tanh))
        return template

    def get_generator(self, z_var):
        return self.generator_template.construct(input=z_var)

    def start_discriminator(self):
        node1_0 = \
            (pt.template("input").
             custom_conv2d(self.df_dim, k_h=4, k_w=4).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 4, k_h=4, k_w=4).
             conv_batch_norm().
             custom_conv2d(self.df_dim * 8, k_h=4, k_w=4).
             conv_batch_norm())
        node1_1 = \
            (node1_0.
             custom_conv2d(self.df_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())

        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(leaky_rectify, leakiness=0.2))

        return node1

    def end_discriminator(self):
        template = \
            (pt.template("input").  # 128*9*4*4
             custom_conv2d(self.df_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1).  # 128*8*4*4
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             # custom_fully_connected(1))
             custom_conv2d(1, k_h=4, k_w=4, d_h=4, d_w=4))

        return template

    def get_discriminator(self, x_var, c_var):
        x_code = self.d_start_template.construct(input=x_var)

        c_code = self.d_context_template.construct(input=c_var)
        # TODO: dhouble check whether tail is correct
        c_code = tf.expand_dims(tf.expand_dims(c_code, 1), 1)
        c_code = tf.tile(c_code, [1, 4, 4, 1])

        x_c_code = tf.concat(3, [x_code, c_code])
        return self.d_end_template.construct(input=x_c_code)
