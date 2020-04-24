import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('vowdata_nohead.dat', dtype='U16')
identifiers = data[:, 0]
data = data[:,1:16].astype(np.int)

vowels = ['ae', 'ah', 'aw', 'eh', 'er', 'ei', 'ih', 'iy', 'oa', 'oo', 'uh', 'uw']


N_train = 70
N_test = 69
N_all = N_train+N_test

train_index = []
test_index = []

for vowel in vowels:
        index = np.flatnonzero(np.core.defchararray.find(identifiers,vowel)!=-1)
        train_index.extend(index[:N_train])
        test_index.extend(index[N_train:])

train_set = data[train_index]
test_set = data[test_index]





ae=data[:N_all, :]
ah=data[N_all:2*N_all, :]
aw=data[2*N_all:3*N_all, :]
eh=data[3*N_all:4*N_all, :]
er=data[4*N_all:5*N_all, :]
ei=data[5*N_all:6*N_all, :]
ih=data[6*N_all:7*N_all, :]
iy=data[7*N_all:8*N_all, :]
oa=data[8*N_all:9*N_all, :]
oo=data[9*N_all:10*N_all, :]
uh=data[10*N_all:11*N_all, :]
uw=data[11*N_all:12*N_all, :]


print(ae)


