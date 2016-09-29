from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np

from infogan.misc.datasets_embedding import FlowerDataset
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset = FlowerDataset()
    dataset_name = 'birds'
    dataset.train = dataset.get_data('../Data/%s/%s64test.pickle' % (dataset_name, dataset_name))
    embeddings = dataset.train._embeddings
    #embeddings = embeddings.reshape([-1, 4800])
    embeddings = np.mean(embeddings,1)
    sample_embeddings = embeddings[0:2000]
    Y_eucldean = pdist(sample_embeddings, metric='euclidean')
    Y_eucldean = squareform(Y_eucldean)
    Y_cos = pdist(sample_embeddings, metric='cosine')
    Y_cos = squareform(Y_cos)
    plt.matshow(Y_eucldean, fignum=1, cmap=plt.cm.gray)
    plt.matshow(Y_cos, fignum=2, cmap=plt.cm.gray)
    plt.show()
