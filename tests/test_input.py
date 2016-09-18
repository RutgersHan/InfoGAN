from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import numpy as np
import tensorflow as tf
sys.path.append('../infogan/')
from misc.datasets_embedding import FlowerDataset


import numpy as np
import pickle
import random



aaa = np.array([i for i in [1,2,3]])
file_name = '/home/han/Documents/CVPR2017/data/flowers/flowers64.pickle'


f_Dataset = FlowerDataset(file_name)
f_Dataset.get_data()



for i in range(2):
    a,b,c = f_Dataset.train.next_batch(3)

print('sss')
