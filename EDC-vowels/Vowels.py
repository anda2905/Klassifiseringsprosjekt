import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

data = np.genfromtxt('vowdata_nohead.dat', dtype='U16')
identifiers = data[:, 0]
data = data[:, 2:7].astype(np.int)  # pulls out 'steady state' values for the vowels


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

ae_training = data[:N_train, :]
ae_test = data[N_train:N_all, :]
ah_training = data[N_all:N_all + N_train, :]
ah_test = data[N_all + N_train:2 * N_all, :]
aw_training = data[2 * N_all:2 * N_all + N_train, :]
aw_test = data[2 * N_all + N_train:3 * N_all, :]
eh_training = data[3 * N_all:3 * N_all + N_train, :]
eh_test = data[3 * N_all + N_train:4 * N_all, :]
er_training = data[4 * N_all:4 * N_all + N_train, :]
er_test = data[4 * N_all + N_train:5 * N_all, :]
ei_training = data[5 * N_all:5 * N_all + N_train, :]
ei_test = data[5 * N_all + N_train:6 * N_all, :]
ih_training = data[6 * N_all:6 * N_all + N_train, :]
ih_test = data[6 * N_all + N_train:7 * N_all, :]
iy_training = data[7 * N_all:7 * N_all + N_train, :]
iy_test = data[7 * N_all + N_train:8 * N_all, :]
oa_training = data[8 * N_all:8 * N_all + N_train, :]
oa_test = data[8 * N_all + N_train:9 * N_all, :]
oo_training = data[9 * N_all:9 * N_all + N_train, :]
oo_test = data[9 * N_all + N_train:10 * N_all, :]
uh_training = data[10 * N_all:10 * N_all + N_train, :]
uh_test = data[10 * N_all + N_train:11 * N_all, :]
uw_training = data[11 * N_all:11 * N_all + N_train, :]
uw_test = data[11 * N_all + N_train:12 * N_all, :]

vowels_train = dict()
vowels_train[0]=ae_training
vowels_train[1]=ah_training
vowels_train[2]=aw_training
vowels_train[3]=eh_training
vowels_train[4]=er_training
vowels_train[5]=ei_training
vowels_train[6]=ih_training
vowels_train[7]=iy_training
vowels_train[8]=oa_training
vowels_train[9]=oo_training
vowels_train[10]=uh_training
vowels_train[11]=uw_training

vowels_test = []
vowels_test.extend(ae_test)
vowels_test.extend(ah_test)
vowels_test.extend(aw_test)
vowels_test.extend(eh_test)
vowels_test.extend(er_test)
vowels_test.extend(ei_test)
vowels_test.extend(ih_test)
vowels_test.extend(iy_test)
vowels_test.extend(oa_test)
vowels_test.extend(oo_test)
vowels_test.extend(uh_test)
vowels_test.extend(uw_test)

vowels_test = [ae_test, ah_test, aw_test, eh_test, er_test, ei_test, ih_test, iy_test, oa_test, oo_test, uh_test,
               uw_test]


def cov_matrix(m):
    return np.cov(m.T)


# print("covariance matrix for ae:\n", cov_matrix(ae))
# print("covariance matrix for ah:\n", cov_matrix(ah))
# print("covariance matrix for aw:\n", cov_matrix(aw))
# print("covariance matrix for eh:\n", cov_matrix(eh))
# print("covariance matrix for er:\n", cov_matrix(er))
# print("covariance matrix for ei:\n", cov_matrix(ei))
# print("covariance matrix for ih:\n", cov_matrix(ih))
# print("covariance matrix for iy:\n", cov_matrix(iy))
# print("covariance matrix for oa:\n", cov_matrix(oa))
# print("covariance matrix for oo:\n", cov_matrix(oo))
# print("covariance matrix for uh:\n", cov_matrix(uh))
# print("covariance matrix for uw:\n", cov_matrix(uw))

def mean(m):
    mean = m.mean(axis=0)
    return mean


def cov(m):
    c = np.dot((m - m.mean(axis=0)).T, (m - m.mean(axis=0))) / (len(m) - 1)
    return c


def gaussian_density(x,m):
    mu=mean(m)
    sigma=cov(m)
    a=np.exp((-1/2)*(x-mu).T*np.linalg.inv(sigma) * (x - mu))
    b=((2*np.pi**len(x)*np.linalg.det(sigma**(1)))**(1/2))

    g_density = a / b
    return g_density




#predict = []
#for vowel in vowels.enumarate:
 #   for i in range(1,70):
  #      g=gaussian_density(vowel[i],vowel)
   #     m=argmax(g)
    #    predict[i]=g









correct_train = np.asarray([i for i in range(12) for _ in range(70)])
correct_test = np.asarray([i for i in range(12) for _ in range(69)])

g = gaussian_density(ae[67], ae)
y = multivariate_normal.pdf(ae, mean(ae), cov(ae))


g=gaussian_density(oa[50],oa)
b=gaussian_density(oa[50],data)

print("ae: ",g)
print("argmax ae:   ",np.argmax(g,axis=0))

print("   ")
print("data    : ",g)
print("argmax data   : ",np.argmax(b,axis=0))
#print(cov(ae))



print("mean data  ",mean(data))
print("cov data   ",cov(data))

#print("mean ae  ",mean(ae))
#print("cov ae   ",cov(ae))

#print("mean data  ",mean(data))
#print("cov data   ",cov(data))


print("  ")

#print(class_probabilities)

#rv = multivariate_normal(ae,mean(ae), cov(ae))


#print(rv.pdf(test_set))
distributions = dict()

distributions[0]=multivariate_normal(mean(ae),cov(ae))
distributions[1]=multivariate_normal(mean(ah),cov(ah))
distributions[2]=multivariate_normal(mean(aw),cov(aw))
distributions[3]=multivariate_normal(mean(eh),cov(eh))
distributions[4]=multivariate_normal(mean(er),cov(er))
distributions[5]=multivariate_normal(mean(ei),cov(ei))
distributions[6]=multivariate_normal(mean(ih),cov(ih))
distributions[7]=multivariate_normal(mean(iy),cov(iy))
distributions[8]=multivariate_normal(mean(oa),cov(oa))
distributions[9]=multivariate_normal(mean(oo),cov(oo))
distributions[10]=multivariate_normal(mean(uh),cov(uh))
distributions[11]=multivariate_normal(mean(uw),cov(uw))

print("distributions:    ",distributions)

vowel_probabilities_test = np.zeros((len(vowels), len(test_set)))

vowel_probabilities_test[0] = distributions[0].pdf(test_set)
vowel_probabilities_test[1] = distributions[1].pdf(test_set)
vowel_probabilities_test[2] = distributions[2].pdf(test_set)
vowel_probabilities_test[3] = distributions[3].pdf(test_set)
vowel_probabilities_test[4] = distributions[4].pdf(test_set)
vowel_probabilities_test[5] = distributions[5].pdf(test_set)
vowel_probabilities_test[6] = distributions[6].pdf(test_set)
vowel_probabilities_test[7] = distributions[7].pdf(test_set)
vowel_probabilities_test[8] = distributions[8].pdf(test_set)
vowel_probabilities_test[9] = distributions[9].pdf(test_set)
vowel_probabilities_test[10] = distributions[10].pdf(test_set)
vowel_probabilities_test[11] = distributions[11].pdf(test_set)




print("vowel_probabilities_test:    ",vowel_probabilities_test)

predicted_vowel_indices_test = np.argmax(vowel_probabilities_test, axis=0)

print("predicted:   ",predicted_vowel_indices_test)


print(len(test_set))
print(len(train_set))