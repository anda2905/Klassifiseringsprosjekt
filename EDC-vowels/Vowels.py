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


def mean(m):
    mn = m.mean(axis=0)
    return mn


print("ae mean:", mean(ae))
print("ah mean:", mean(ah))
print("aw mean:", mean(aw))
print("eh mean:", mean(eh))
print("er mean:", mean(er))
print("ei mean:", mean(ei))
print("ih mean:", mean(ih))
print("iy mean:", mean(iy))
print("oa mean:", mean(oa))
print("oo mean:", mean(oo))
print("uh mean:", mean(uh))
print("uw mean:", mean(uw))

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


def cov(m):
    c = np.dot((m - m.mean(axis=0)).T, (m - m.mean(axis=0))) / (len(m) - 1)
    return c


print("covae2:", cov(ae))

rv = multivariate_normal(mean=mean(ae), cov=cov(ae))

# print("f: ")
# print(cov_ae[1])

print("   ")
print("   ")
print(rv)
