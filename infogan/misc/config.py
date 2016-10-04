from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: flowers, faces, birds
__C.DATASET_NAME = 'birds'
# Encoder input: text, attribute
__C.ENCODER_INPUT = 'attribute'
__C.FILENAME = 'birds64image_mask_bg_attr_text'
__C.CONFIG_NAME = ''
__C.GPU_ID = 0

# Training options
__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.BATCH_SIZE = 256
__C.TRAIN.NUM_COPY = 8
__C.TRAIN.UPDATES_PER_EPOCH = 50
__C.TRAIN.MAX_EPOCH = 2000
__C.TRAIN.SNAPSHOT_INTERVAL = 2000
__C.TRAIN.PRETRAINED_MODEL = ''

__C.TRAIN.BG_DISCRIMINATOR_LR = 2e-4
__C.TRAIN.FG_DISCRIMINATOR_LR = 2e-4
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.ENCODER_LR = 2e-4
__C.TRAIN.MASK_FLAG = False

__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.REG_Z = 0.0
__C.TRAIN.COEFF.FAKE_C = 0.0
__C.TRAIN.COEFF.NOISE_C = 0.0
__C.TRAIN.COEFF.REAL_C = 0.0
__C.TRAIN.COEFF.LIKE = 0.0

# Modal options
__C.GAN = edict()
__C.GAN.EMBEDDING_DIM = 312
__C.GAN.NETWORK_TYPE = 'default'

__C.GAN.LATENT_SPEC = edict()
__C.GAN.LATENT_SPEC.UNIFORM_FLAG = False
__C.GAN.LATENT_SPEC.UNIFORM = edict()
__C.GAN.LATENT_SPEC.UNIFORM.REG = False
__C.GAN.LATENT_SPEC.UNIFORM.DIM = -1

__C.GAN.LATENT_SPEC.GAUSSIAN_FLAG = False
__C.GAN.LATENT_SPEC.GAUSSIAN = edict()
__C.GAN.LATENT_SPEC.GAUSSIAN.REG = False
__C.GAN.LATENT_SPEC.GAUSSIAN.DIM = -1

__C.GAN.LATENT_SPEC.CAT_FLAG = False
__C.GAN.LATENT_SPEC.CAT = edict()
__C.GAN.LATENT_SPEC.CAT.REG = False
__C.GAN.LATENT_SPEC.CAT.DIM = -1

__C.GAN.LATENT_SPEC.BERNOULLI_FLAG = False
__C.GAN.LATENT_SPEC.BERNOULLI = edict()
__C.GAN.LATENT_SPEC.BERNOULLI.REG = False
__C.GAN.LATENT_SPEC.BERNOULLI.DIM = -1

__C.GAN.CON_LATENT_SPEC = edict()
__C.GAN.CON_LATENT_SPEC.UNIFORM_FLAG = False
__C.GAN.CON_LATENT_SPEC.UNIFORM = edict()
__C.GAN.CON_LATENT_SPEC.UNIFORM.DIM = -1

__C.GAN.CON_LATENT_SPEC.GAUSSIAN_FLAG = False
__C.GAN.CON_LATENT_SPEC.GAUSSIAN = edict()
__C.GAN.CON_LATENT_SPEC.GAUSSIAN.DIM = -1

__C.GAN.CON_LATENT_SPEC.CAT_FLAG = False
__C.GAN.CON_LATENT_SPEC.CAT = edict()
__C.GAN.CON_LATENT_SPEC.CAT.DIM = -1

__C.GAN.CON_LATENT_SPEC.BERNOULLI_FLAG = False
__C.GAN.CON_LATENT_SPEC.BERNOULLI = edict()
__C.GAN.CON_LATENT_SPEC.BERNOULLI.DIM = -1


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
