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
        if cfg.GAN.NETWORK_TYPE == "default":
            with tf.variable_scope("fg_d_net"):
                self.fg_shared_template = self.discriminator_shared()
                self.fg_discriminator_notshared_template = self.discriminator_notshared()
            with tf.variable_scope("fg_g_net"):
                self.fg_generator_template = self.generator()
            with tf.variable_scope("fg_e_net"):
                self.fg_context_template = self.context_embedding()
        else:
            raise NotImplementedError

    def context_embedding(self):
        template = (pt.template("input").
                    custom_fully_connected(self.ef_dim * 4).
                    fc_batch_norm().
                    apply(tf.nn.relu).
                    custom_fully_connected(self.ef_dim * 2))
        return template

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

    def generator(self):
        s = self.image_size
        s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(s / 16)
        generator_template = \
            (pt.template("input").
             custom_fully_connected(s16 * s16 * self.gf_dim * 8).
             fc_batch_norm().
             apply(tf.nn.relu).
             reshape([-1, s16, s16, self.gf_dim * 8]).
             custom_deconv2d([0, s8, s8, self.gf_dim * 4], k_h=4, k_w=4).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, s4, s4, self.gf_dim * 2], k_h=4, k_w=4).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, s2, s2, self.gf_dim], k_h=4, k_w=4).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0] + list(self.image_shape), k_h=4, k_w=4).
             apply(tf.nn.tanh))
        return generator_template

    def generate_fg_condition(self, c_var):
        conditions = self.fg_context_template.construct(input=c_var)
        mean = conditions[:, :self.ef_dim]
        log_sigma = conditions[:, self.ef_dim:]
        condition_list = [mean, log_sigma]
        return condition_list

    def get_fg_discriminator_shared(self, x_var):
        return self.fg_shared_template.construct(input=x_var)

    def get_fg_discriminator(self, shared_layers):
        return self.fg_discriminator_notshared_template.construct(input=shared_layers)

    def get_fg_generator(self, z_var):
        return self.fg_generator_template.construct(input=z_var)





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
