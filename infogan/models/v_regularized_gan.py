from infogan.misc.distributions import Product, Distribution, Gaussian, Categorical, Bernoulli
import prettytensor as pt
import tensorflow as tf
import infogan.misc.custom_ops
from infogan.misc.custom_ops import leaky_rectify
from infogan.misc.config import cfg

#  TODO: In discriminate
#  Does template.constrct really shared the computation, I did 3 times construct


class ConRegularizedGAN(object):
    def __init__(self, output_dist, latent_spec, con_latent_spec,
                 image_shape, gf_dim=64, df_dim=64):
        """
        :type output_dist: Distribution e.g. MeanBernoulli(dataset.image_dim),
        :type latent_spec: list[(Distribution, bool)]
                    e.g.    latent_spec = [
                            (Uniform(62), False),
                            (Categorical(10), True),
                            (Uniform(1, fix_std=True), True),
                            (Uniform(1, fix_std=True), True),
                        ]
        :type con_latent_spec:  (LatentGaussian, (ef_dim, fix_std=True), True)
        :type batch_size: int
        :type network_type: string
        """
        self.output_dist = output_dist
        self.latent_spec = latent_spec
        self.latent_dist = Product([x for x, _ in latent_spec])
        self.con_latent_spec = con_latent_spec
        self.con_latent_dist = Product([x for x in con_latent_spec])
        self.reg_latent_dist = Product([x for x, reg in latent_spec if reg])
        self.nonreg_latent_dist = Product([x for x, reg in latent_spec if not reg])
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.network_type = cfg.GAN.NETWORK_TYPE
        self.image_shape = image_shape
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        assert all(isinstance(x, (Gaussian, Categorical, Bernoulli)) for x in self.reg_latent_dist.dists)

        self.reg_cont_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, Gaussian)])
        self.reg_disc_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, (Categorical, Bernoulli))])

        self.image_size = image_shape[0]
        self.image_shape = image_shape
        self.s = self.image_size
        self.s2, self.s4, self.s8, self.s16 = int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)
        if cfg.GAN.NETWORK_TYPE == "default":
            with tf.variable_scope("d_net"):
                self.shared_template = self.discriminator_shared()
                self.discriminator_notshared_template = self.discriminator_notshared()
            with tf.variable_scope("g_net"):
                self.fg_template = self.fg_generator()
                self.bg_template = self.bg_generator()
                self.generator_template = self.generator()
        else:
            raise NotImplementedError

    def fg_generator(self):
        template = (pt.template("input").
                    custom_conv2d(self.df_dim, k_h=5, k_w=5).  # -->32*32*64
                    conv_batch_norm().
                    apply(leaky_rectify).
                    custom_conv2d(self.df_dim * 2, k_h=5, k_w=5).  # -->16*16*128
                    conv_batch_norm().
                    apply(leaky_rectify))
        return template

    def bg_generator(self):
        template = (pt.template("input").
                    custom_fully_connected(self.s16 * self.s16 * self.gf_dim * 4).
                    fc_batch_norm().
                    apply(tf.nn.relu).
                    reshape([-1, self.s16, self.s16, self.gf_dim * 4]).  # -->4*4*256
                    custom_deconv2d([0, self.s8, self.s8, self.gf_dim * 2], k_h=4, k_w=4).  # -->8*8*128
                    conv_batch_norm().
                    apply(tf.nn.relu).
                    custom_deconv2d([0, self.s4, self.s4, self.gf_dim], k_h=4, k_w=4).  # -->16*16*64
                    conv_batch_norm().
                    apply(tf.nn.relu))
        return template

    def generator(self):
        generator_template = \
            (pt.template("input").
             custom_conv2d(self.df_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).  # -->16*16*256
             conv_batch_norm().
             apply(leaky_rectify).
             custom_deconv2d([0, self.s2, self.s2, self.gf_dim * 2], k_h=4, k_w=4).  # -->32*32*128
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s, self.s, self.gf_dim], k_h=4, k_w=4).  # -->64*64*64
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0] + list(self.image_shape), k_h=5, k_w=5, d_h=1, d_w=1).
             apply(tf.nn.tanh))
        return generator_template

    def get_generator(self, x_var, z_var):
        fg_encode = self.fg_template.construct(input=x_var)
        bg_decode = self.bg_template.construct(input=z_var)
        fg_bg_var = tf.concat(3, [fg_encode, bg_decode])
        return self.generator_template.construct(input=fg_bg_var)

    def discriminator_shared(self):
        last_conv_template = \
            (pt.template("input").
             custom_conv2d(self.df_dim, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify).
             custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify).
             custom_conv2d(self.df_dim * 4, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify).
             custom_conv2d(self.df_dim * 8, k_h=4, k_w=4)
             )
        shared_template = last_conv_template
        return shared_template

    def discriminator_notshared(self):
        # ####Use shared_template from discriminator as input
        discriminator_template = \
            (pt.template("input").
             conv_batch_norm().
             apply(leaky_rectify).
             custom_fully_connected(1))

        return discriminator_template

    def get_discriminator_shared(self, x_var):
        return self.shared_template.construct(input=x_var)

    def get_discriminator(self, shared_layers):
        return self.discriminator_notshared_template.construct(input=shared_layers)




    def disc_reg_z(self, reg_z_var):
        ret = []
        for dist_i, z_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_var(reg_z_var)):
            if isinstance(dist_i, (Categorical, Bernoulli)):
                ret.append(z_i)
        return self.reg_disc_latent_dist.join_vars(ret)

    def cont_reg_z(self, reg_z_var):
        ret = []
        for dist_i, z_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_var(reg_z_var)):
            if isinstance(dist_i, Gaussian):
                ret.append(z_i)
        return self.reg_cont_latent_dist.join_vars(ret)

    def disc_reg_dist_info(self, reg_dist_info):
        ret = []
        for dist_i, dist_info_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_dist_info(reg_dist_info)):
            if isinstance(dist_i, (Categorical, Bernoulli)):
                ret.append(dist_info_i)
        return self.reg_disc_latent_dist.join_dist_infos(ret)

    def cont_reg_dist_info(self, reg_dist_info):
        ret = []
        for dist_i, dist_info_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_dist_info(reg_dist_info)):
            if isinstance(dist_i, Gaussian):
                ret.append(dist_info_i)
        return self.reg_cont_latent_dist.join_dist_infos(ret)

    def reg_z(self, z_var):
        ret = []
        for (_, reg_i), z_i in zip(self.latent_spec, self.latent_dist.split_var(z_var)):
            if reg_i:
                ret.append(z_i)
        return self.reg_latent_dist.join_vars(ret)

    def nonreg_z(self, z_var):
        ret = []
        for (_, reg_i), z_i in zip(self.latent_spec, self.latent_dist.split_var(z_var)):
            if not reg_i:
                ret.append(z_i)
        return self.nonreg_latent_dist.join_vars(ret)

    def reg_dist_info(self, dist_info):
        ret = []
        for (_, reg_i), dist_info_i in zip(self.latent_spec, self.latent_dist.split_dist_info(dist_info)):
            if reg_i:
                ret.append(dist_info_i)
        return self.reg_latent_dist.join_dist_infos(ret)

    def nonreg_dist_info(self, dist_info):
        ret = []
        for (_, reg_i), dist_info_i in zip(self.latent_spec, self.latent_dist.split_dist_info(dist_info)):
            if not reg_i:
                ret.append(dist_info_i)
        return self.nonreg_latent_dist.join_dist_infos(ret)

    def combine_reg_nonreg_z(self, reg_z_var, nonreg_z_var):
        reg_z_vars = self.reg_latent_dist.split_var(reg_z_var)
        reg_idx = 0
        nonreg_z_vars = self.nonreg_latent_dist.split_var(nonreg_z_var)
        nonreg_idx = 0
        ret = []
        for idx, (dist_i, reg_i) in enumerate(self.latent_spec):
            if reg_i:
                ret.append(reg_z_vars[reg_idx])
                reg_idx += 1
            else:
                ret.append(nonreg_z_vars[nonreg_idx])
                nonreg_idx += 1
        return self.latent_dist.join_vars(ret)

    def combine_reg_nonreg_dist_info(self, reg_dist_info, nonreg_dist_info):
        reg_dist_infos = self.reg_latent_dist.split_dist_info(reg_dist_info)
        reg_idx = 0
        nonreg_dist_infos = self.nonreg_latent_dist.split_dist_info(nonreg_dist_info)
        nonreg_idx = 0
        ret = []
        for idx, (dist_i, reg_i) in enumerate(self.latent_spec):
            if reg_i:
                ret.append(reg_dist_infos[reg_idx])
                reg_idx += 1
            else:
                ret.append(nonreg_dist_infos[nonreg_idx])
                nonreg_idx += 1
        return self.latent_dist.join_dist_infos(ret)
