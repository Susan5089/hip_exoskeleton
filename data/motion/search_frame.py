import numpy as np 
from sklearn.metrics import pairwise_distances
data = np.loadtxt("/home/shuzhen/Exo_human_walk_test4/data/motion/NJIT_BME_tmp.mot")
data = data[:, 8:]

distance = pairwise_distances(np.expand_dims(data[-1], 0), data[:(data.shape[0])-10], metric='euclidean') 