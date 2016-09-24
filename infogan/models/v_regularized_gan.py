from infogan.misc.distributions import Product, Distribution, Gaussian, Categorical, Bernoulli
import prettytensor as pt
import tensorflow as tf
import infogan.misc.custom_ops
from infogan.misc.custom_ops import leaky_rectify

#  TODO: In discriminate
#  Does template.constrct really shared the computation, I did 3 times construct

class ConRegularizedGAN(object):
    def __init__(self, output_dist, latent_spec, con_latent_spec, batch_size,
                 image_shape, network_type, gf_dim=64, df_dim=64, ef_dim=1024):
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
        self.con_latent_dist = Product([x for x, _ in con_latent_spec])
        self.reg_latent_dist = Product([x for x, reg in latent_spec if reg])
        self.nonreg_latent_dist = Product([x for x, reg in latent_spec if not reg])
        self.batch_size = batch_size
        self.network_type = network_type
        self.image_shape = image_shape
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.ef_dim = ef_dim
        assert all(isinstance(x, (Gaussian, Categorical, Bernoulli)) for x in self.reg_latent_dist.dists)

        self.reg_cont_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, Gaussian)])
        self.reg_disc_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, (Categorical, Bernoulli))])

        self.image_size = image_shape[0]
        self.image_shape = image_shape
        if network_type == "flower":
            with tf.variable_scope("d_net"):
                d_template, e_template, con_template, feature_template = self.discriminator()
                self.discriminator_template = d_template
                self.encoder_template = e_template
                self.context_encoder_template = con_template
                self.feature_template = feature_template
            with tf.variable_scope("g_net"):
                self.generator_template = self.generator()
            with tf.variable_scope("c_net"):
                self.context_template = self.context_embedding()
        else:
            raise NotImplementedError

    def context_embedding(self):
        template = (pt.template("input").
                    custom_fully_connected(self.ef_dim * 2).
                    fc_batch_norm().
                    apply(tf.nn.relu).
                    custom_fully_connected(self.ef_dim * 2))
        return template

    def discriminator(self):
        feature_template = \
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
        shared_template = \
            (feature_template.
             conv_batch_norm().
             apply(leaky_rectify))
        discriminator_template = shared_template.custom_fully_connected(1)
        if self.reg_latent_dist.dist_flat_dim > 0:
            encoder_template = \
                (shared_template.
                 custom_fully_connected(128).
                 fc_batch_norm().
                 apply(leaky_rectify).
                 custom_fully_connected(self.reg_latent_dist.dist_flat_dim))
        else:
            encoder_template = None
        context_encoder_template = \
            (shared_template.
             custom_fully_connected(self.ef_dim * 2).
             fc_batch_norm().
             apply(leaky_rectify).
             # self.con_latent_dist.dist_flat_dim = 2 * ef_dim for gaussian
             custom_fully_connected(self.ef_dim))
        return (discriminator_template, encoder_template,
                context_encoder_template, feature_template)

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
             custom_deconv2d([0] + list(self.image_shape), k_h=4, k_w=4))
        return generator_template

    def discriminate_old(self, x_var):
        d_out = self.discriminator_template.construct(input=x_var)
        d = tf.nn.sigmoid(d_out[:, 0])
        reg_dist_flat = self.encoder_template.construct(input=x_var)
        reg_dist_info = self.reg_latent_dist.activate_dist(reg_dist_flat)
        return d, self.reg_latent_dist.sample(reg_dist_info), reg_dist_info, reg_dist_flat

    def generate_old(self, z_var):
        x_dist_flat = self.generator_template.construct(input=z_var)
        x_dist_info = self.output_dist.activate_dist(x_dist_flat)
        return self.output_dist.sample(x_dist_info), x_dist_info

    """
    def discriminate(self, x_var):
        d_out = self.discriminator_template.construct(input=x_var)
        d = tf.nn.sigmoid(d_out[:, 0])
        con_dist_flat = self.context_encoder_template.construct(input=x_var)
        con_dist_info = self.con_latent_dist.activate_dist(con_dist_flat)
        if self.reg_latent_dist.dist_flat_dim > 0:
            reg_dist_flat = self.encoder_template.construct(input=x_var)
            reg_dist_info = self.reg_latent_dist.activate_dist(reg_dist_flat)
            return (d, self.reg_latent_dist.sample(reg_dist_info), reg_dist_info,
                    reg_dist_flat, self.con_latent_dist.sample(con_dist_info),
                    con_dist_info, con_dist_flat)
        else:
            reg_dist_flat = None
            reg_dist_info = None
            return (d, None, reg_dist_info,
                    reg_dist_flat, self.con_latent_dist.sample(con_dist_info),
                    con_dist_info, con_dist_flat)
    """
    def discriminate(self, x_var):
        d_out = self.discriminator_template.construct(input=x_var)
        f_out = self.feature_template.construct(input=x_var)
        e_out = self.context_encoder_template.construct(input=x_var)
        return d_out, f_out, e_out

    def generate(self, z_var):
        x_dist = self.generator_template.construct(input=z_var)
        return tf.nn.tanh(x_dist)

    def generate_for_visualization(self, image_num, embedding_shape):
        embeddings = tf.placeholder(
            tf.float32, [None] + embedding_shape,
            name='conditional_embeddings'
        )
        z_var = self.latent_dist.sample_prior(image_num)
        c_var = self.generate_condition(embeddings)
        z_c_var = tf.concat(1, [z_var, c_var])
        generated_images = self.generate(z_c_var)
        return embeddings, generated_images

    def generate_condition(self, c_var):
        conditions = self.context_template.construct(input=c_var)
        mean = conditions[:, :self.ef_dim]
        log_sigma = conditions[:, self.ef_dim:]
        condition_list = [mean, log_sigma]
        return condition_list

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
