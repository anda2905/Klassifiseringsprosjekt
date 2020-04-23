import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('vowdata_nohead.dat', dtype='U16')
identifiers = data[:, 0]
data = data[:,1:16].astype(np.int)
print(data)
print(identifiers)

#vowel (ae="had", ah="hod", aw="hawed", eh="head", er="heard",
 #                       ei="haid", ih="hid", iy="heed", oa=/o/ as in "boat",
  #                      oo="hood", uh="hud", uw="who'd")


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
