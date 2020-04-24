import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('vowdata_nohead.dat', dtype='U16')
identifiers = data[:, 0]
data = data[:,2:7].astype(np.int)

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

ae_mean = ae.mean()
ah_mean = ah.mean()
aw_mean = aw.mean()
eh_mean = eh.mean()
er_mean = er.mean()
ei_mean = ei.mean()
ih_mean = ih.mean()
iy_mean = iy.mean()
oa_mean = oa.mean()
oo_mean = oo.mean()
uh_mean = uh.mean()
uw_mean = uw.mean()

print("ae mean:", ae_mean)
print("ah mean:", ah_mean)
print("aw mean:", aw_mean)
print("eh mean:", eh_mean)
print("er mean:", er_mean)
print("ei mean:", ei_mean)
print("ih mean:", ih_mean)
print("iy mean:", iy_mean)
print("oa mean:", oa_mean)
print("oo mean:", oo_mean)
print("uh mean:", uh_mean)
print("uw mean:", uw_mean)

def cov_matrix(m):
        return np.cov(m.T)

print("covariance matrix for ae:\n", cov_matrix(ae))
print("covariance matrix for ah:\n", cov_matrix(ah))
print("covariance matrix for aw:\n", cov_matrix(aw))
print("covariance matrix for eh:\n", cov_matrix(eh))
print("covariance matrix for er:\n", cov_matrix(er))
print("covariance matrix for ei:\n", cov_matrix(ei))
print("covariance matrix for ih:\n", cov_matrix(ih))
print("covariance matrix for iy:\n", cov_matrix(iy))
print("covariance matrix for oa:\n", cov_matrix(oa))
print("covariance matrix for oo:\n", cov_matrix(oo))
print("covariance matrix for uh:\n", cov_matrix(uh))
print("covariance matrix for uw:\n", cov_matrix(uw))






