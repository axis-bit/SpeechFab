from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

__C.DATASET_NAME = 'shape'
__C.CONFIG_NAME = ''
__C.DATA_DIR = ''
__C.GPU_ID = ''
__C.WORKERS = 8
__C.SHAPE_SIZE = 16
__C.CHANNEL = 1
__C.OUTPUT = '.'
__C.RANDOM_SEED = False
__C.SPLIT = 'train'
__C.CATEGORY = 3
__C.WORDS_NUM = 100

__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.MAX_EPOCH = 3
__C.TRAIN.SNAPSHOT_INTERVAL = 5
__C.TRAIN.DISCRIMINATOR_LR = 0.0001
__C.TRAIN.GENERATOR_LR = 0.0001
__C.TRAIN.ENCODER_LR = 0.0001
__C.TRAIN.N_CRITIC = 5
__C.TRAIN.GANMMA = 10
__C.TRAIN.NET_G = ''
__C.TRAIN.NET_E = ''
__C.TRAIN.CLIP_VALUE = 0.01
__C.TRAIN.RNN_GRAD_CLIP = 0.25

__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 5.0
__C.TRAIN.SMOOTH.GAMMA3 = 10.0
__C.TRAIN.SMOOTH.GAMMA2 = 5.0
__C.TRAIN.SMOOTH.LAMBDA = 1.0


__C.GAN = edict()
__C.GAN.Z_DIM = 128

__C.TEXT = edict()
__C.TEXT.WORDS_NUM = 160
__C.TEXT.EMBEDDING_DIM = 128


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b:
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
        yaml_cfg = edict(yaml.safe_load(f))

    _merge_a_into_b(yaml_cfg, __C)