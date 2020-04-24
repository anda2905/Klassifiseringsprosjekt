import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

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

vowels_train =[ae_training,ah_training,aw_training,eh_training,er_training,ei_training,ih_training,iy_training,oa_training,oo_training,uh_training,uw_training]
vowels_test = [ae_test,ah_test,aw_test,eh_test,er_test,ei_test,ih_test,iy_test,oa_test,oo_test,uh_test,uw_test]

#ae_mean = ae.mean(axis=0)
#ah_mean = ah.mean(axis=0)
#aw_mean = aw.mean(axis=0)
#eh_mean = eh.mean(axis=0)
#er_mean = er.mean(axis=0)
#ei_mean = ei.mean(axis=0)
#ih_mean = ih.mean(axis=0)
#iy_mean = iy.mean(axis=0)
#oa_mean = oa.mean(axis=0)
#oo_mean = oo.mean(axis=0)
#uh_mean = uh.mean(axis=0)
#uw_mean = uw.mean(axis=0)

#print("ae mean:", ae_mean)
#print("ah mean:", ah_mean)
#print("aw mean:", aw_mean)
#print("eh mean:", eh_mean)
#print("er mean:", er_mean)
#print("ei mean:", ei_mean)
#print("ih mean:", ih_mean)
#print("iy mean:", iy_mean)
#print("oa mean:", oa_mean)
#print("oo mean:", oo_mean)
#print("uh mean:", uh_mean)
#print("uw mean:", uw_mean)

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
        c = np.dot((m-m.mean(axis=0)).T,(m-m.mean(axis=0)))/(len(m)-1)
        return c

#print("covae2:",cov(ae))

sample_mean = []
covarianse = []
probabilities_train = []
probabilities_test = []

for i in range (0,12):
    m_train = vowels_train[i]
    m_test = vowels_test[i]
    sample_mean.append(mean(m_train))
    covarianse.append(cov(m_train))
    rv_train = multivariate_normal(mean=mean(m_train),cov=cov(m_train))
    rv_test = multivariate_normal(mean=mean(m_test),cov=cov(m_test))
    probabilities_train.append(rv_train.pdf(m_train))
    probabilities_test.append(rv_train.pdf(m_test))

predicted_train = np.argmax(probabilities_train, axis=0)
predicted_test = np.argmax(probabilities_test, axis=0)

#print("f: ")
#print(cov_ae[1])


print("probabilties training: ",probabilities_train)
print("  ")
print("probabilities test: ",probabilities_test)
print("   ")
print("probabilties training sixe: ",len(probabilities_train))
print("  ")
print("probabilities test size: ",len(probabilities_test))
print("   ")
print("predicted train: ",predicted_train)
print("  ")
print("predicted test: ",predicted_test)
print("   ")
print("predicted train length: ",len(predicted_train))
print("  ")
print("predicted test length: ",len(predicted_test))



#print(mean)
#print(covarianse)




