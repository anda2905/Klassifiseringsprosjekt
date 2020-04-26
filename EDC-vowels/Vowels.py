import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

data = np.genfromtxt('vowdata_nohead.dat', dtype='U16')
identifiers = data[:, 0]
data = data[:, 2:7].astype(np.int)

vowels = ['ae', 'ah', 'aw', 'eh', 'er', 'ei', 'ih', 'iy', 'oa', 'oo', 'uh', 'uw']


N_train = 70
N_test = 69
N_all = N_train + N_test

train_index = []
test_index = []

for vowel in vowels:
    index = np.flatnonzero(np.core.defchararray.find(identifiers, vowel) != -1)
    train_index.extend(index[:N_train])
    test_index.extend(index[N_train:])

train_set = data[train_index]
test_set = data[test_index]

ae = data[:N_all, :]
ah = data[N_all:2 * N_all, :]
aw = data[2 * N_all:3 * N_all, :]
eh = data[3 * N_all:4 * N_all, :]
er = data[4 * N_all:5 * N_all, :]
ei = data[5 * N_all:6 * N_all, :]
ih = data[6 * N_all:7 * N_all, :]
iy = data[7 * N_all:8 * N_all, :]
oa = data[8 * N_all:9 * N_all, :]
oo = data[9 * N_all:10 * N_all, :]
uh = data[10 * N_all:11 * N_all, :]
uw = data[11 * N_all:12 * N_all, :]

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

vowels_train = []
vowels_train.append(ae_training)
vowels_train.append(ah_training)
vowels_train.append(aw_training)
vowels_train.append(eh_training)
vowels_train.append(er_training)
vowels_train.append(ei_training)
vowels_train.append(ih_training)
vowels_train.append(iy_training)
vowels_train.append(oa_training)
vowels_train.append(oo_training)
vowels_train.append(uh_training)
vowels_train.append(uw_training)

vowels_test = []
vowels_test.append(ae_test)
vowels_test.append(ah_test)
vowels_test.append(aw_test)
vowels_test.append(eh_test)
vowels_test.append(er_test)
vowels_test.append(ei_test)
vowels_test.append(ih_test)
vowels_test.append(iy_test)
vowels_test.append(oa_test)
vowels_test.append(oo_test)
vowels_test.append(uh_test)
vowels_test.append(uw_test)

vowels_test = [ae_test,ah_test,aw_test,eh_test,er_test,ei_test,ih_test,iy_test,oa_test,oo_test,uh_test,uw_test]



def cov_matrix(m):
    return np.cov(m.T)

#print("covariance matrix for ae:\n", cov_matrix(ae))
#print("covariance matrix for ah:\n", cov_matrix(ah))
#print("covariance matrix for aw:\n", cov_matrix(aw))
#print("covariance matrix for eh:\n", cov_matrix(eh))
#print("covariance matrix for er:\n", cov_matrix(er))
#print("covariance matrix for ei:\n", cov_matrix(ei))
#print("covariance matrix for ih:\n", cov_matrix(ih))
#print("covariance matrix for iy:\n", cov_matrix(iy))
#print("covariance matrix for oa:\n", cov_matrix(oa))
#print("covariance matrix for oo:\n", cov_matrix(oo))
#print("covariance matrix for uh:\n", cov_matrix(uh))
#print("covariance matrix for uw:\n", cov_matrix(uw))

def mean(m):
    mean=m.mean(axis=0)
    return mean

def cov(m):
    c = np.dot((m - m.mean(axis=0)).T, (m - m.mean(axis=0))) / (len(m) - 1)
    return c


def gaussian_density(x,mu,sigma):
    a=np.exp((-1/2)*(x-mu).T*np.linalg.inv(sigma) * (x - mu))
    b=((2*np.pi**len(x)*np.linalg.det(sigma**(1)))**(1/2))

    g_density=a/b
    return g_density










correct_train = np.asarray([i for i in range(12) for _ in range(70)])
correct_test = np.asarray([i for i in range(12) for _ in range(69)])



g=gaussian_density(ae[0],mean(ae),cov(ae))

print("dobbelsjekk: ",g)



print(cov(ae))




