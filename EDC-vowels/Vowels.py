import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('vowdata_nohead.dat', dtype='U16')
identifiers = data[:, 0]
data = data[:,1:16].astype(np.int)



N_train = 70
N_test = 69
N_all = N_train+N_test

ae_training=data[:N_train, :]
ae_test=data[N_train:N_all,:]
ah_training=data[N_all:N_all+N_train, :]
ah_test=data[N_all+N_train:2*N_all,:]
aw_training=data[2*N_all:2*N_all + N_train, :]
aw_test=data[2*N_all + N_train:3*N_all,:]
eh_training=data[3*N_all:3*N_all + N_train, :]
eh_test=data[3*N_all + N_train:4*N_all,:]
er_training=data[4*N_all:4*N_all + N_train, :]
er_test=data[4*N_all + N_train:5*N_all,:]
ei_training=data[5*N_all:5*N_all + N_train, :]
ei_test=data[5*N_all + N_train:6*N_all,:]
ih_training=data[6*N_all:6*N_all + N_train, :]
ih_test=data[6*N_all + N_train:7*N_all,:]
iy_training=data[7*N_all:7*N_all + N_train, :]
iy_test=data[7*N_all + N_train:8*N_all,:]
oa_training=data[8*N_all:8*N_all + N_train, :]
oa_test=data[8*N_all + N_train:9*N_all,:]
oo_training=data[9*N_all:9*N_all + N_train, :]
oo_test=data[9*N_all + N_train:10*N_all,:]
uh_training=data[10*N_all:10*N_all + N_train, :]
uh_test=data[10*N_all + N_train:11*N_all,:]
uw_training=data[11*N_all:11*N_all + N_train, :]
uw_test=data[11*N_all + N_train:12*N_all,:]

print(uw_test)