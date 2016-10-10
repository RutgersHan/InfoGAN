from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np

from infogan.misc.datasets_embedding import TextDataset
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

if __name__ == "__main__":
    datadir = '/home/tao/deep_learning/CVPR2017/icml16/InfoGAN/Data/birds'
    dataset = TextDataset(datadir)
    filename_test = '%s/%s/test' % (datadir, 'pickle')
    dataset.test = dataset.get_data(filename_test)
    images_test, _, _, embeddings_test, captions_test, _, _ = dataset.test.next_batch(64, 1)
    
    print("success")