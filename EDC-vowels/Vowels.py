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


def mean(m):
    mean = m.mean(axis=0)
    return mean


def cov(m):
    d = True
    c = np.dot((m - m.mean(axis=0)).T, (m - m.mean(axis=0))) / (len(m) - 1)
    if d == True:
        c = np.diag(np.diag(c))
    return c


def gaussian_density(x,m):
    mu=mean(m)
    sigma=cov(m)
    a=np.exp((-1/2)*(x-mu).T*np.linalg.inv(sigma) * (x - mu))
    b=((2*np.pi**len(x)*np.linalg.det(sigma**(1)))**(1/2))

    g_density = a / b
    return g_density















correct_train = np.asarray([i for i in range(12) for _ in range(70)])
correct_test = np.asarray([i for i in range(12) for _ in range(69)])

g = gaussian_density(ae[67], ae)
y = multivariate_normal.pdf(ae, mean(ae), cov(ae))




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

vowel_probabilities_train = np.zeros((len(vowels), len(train_set)))

vowel_probabilities_train[0] = distributions[0].pdf(train_set)
vowel_probabilities_train[1] = distributions[1].pdf(train_set)
vowel_probabilities_train[2] = distributions[2].pdf(train_set)
vowel_probabilities_train[3] = distributions[3].pdf(train_set)
vowel_probabilities_train[4] = distributions[4].pdf(train_set)
vowel_probabilities_train[5] = distributions[5].pdf(train_set)
vowel_probabilities_train[6] = distributions[6].pdf(train_set)
vowel_probabilities_train[7] = distributions[7].pdf(train_set)
vowel_probabilities_train[8] = distributions[8].pdf(train_set)
vowel_probabilities_train[9] = distributions[9].pdf(train_set)
vowel_probabilities_train[10] = distributions[10].pdf(train_set)
vowel_probabilities_train[11] = distributions[11].pdf(train_set)


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


#print("vowel_probabilities_train:    ",vowel_probabilities_train)

predicted_vowel_indices_train = np.argmax(vowel_probabilities_train, axis=0)

#print("predicted train:   ",predicted_vowel_indices_train)





#print("vowel_probabilities_test:    ",vowel_probabilities_test)

predicted_vowel_indices_test = np.argmax(vowel_probabilities_test, axis=0)

#print("predicted test:   ",predicted_vowel_indices_test)

def error_rate(N, m):
    e = np.sum(m) - np.trace(m)
    print("#feil:", e)
    return e / N*100

#print(correct_train)
conf_train = np.zeros((12, 12))
# confusion matrix for the train set
for i, j in zip(predicted_vowel_indices_train, correct_train):
    conf_train[i, j] += 1
print('train confusion:\n', conf_train)


print("train feil: ", error_rate(N_train*12, conf_train),"%")
print("  ")

conf_test = np.zeros((12, 12))
# confusion matrix for the train set
for i, j in zip(predicted_vowel_indices_test, correct_test):
    conf_test[i, j] += 1
print('test confusion:\n', conf_test)


print("test feil: ", error_rate(N_test*12, conf_test),"%")