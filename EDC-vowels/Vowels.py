import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import multivariate_normal

# loads data
data = np.genfromtxt('vowdata_nohead.dat', dtype='U16')
identifiers = data[:, 0]
data = data[:, 2:5].astype(np.int)  # pulls out 'steady state' values for the vowels

vowels = ['ae', 'ah', 'aw', 'eh', 'er', 'ei', 'ih', 'iy', 'oa', 'oo', 'uh', 'uw']

# distributes training and testing data
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


# put data into dictionary and creates answersheats
data_dict = dict()
data_dict_train = dict()

for i in range(12):
    data_dict[vowels[i]] = data[i * N_all:(i + 1) * N_all, :]
    data_dict_train[vowels[i]] = data[i * N_all: i * N_all + N_train, :]


correct_train = np.asarray([i for i in range(12) for _ in range(N_train)])
correct_test = np.asarray([i for i in range(12) for _ in range(N_test)])


# defines a function for the mean
def mean(m):
    mean = m.mean(axis=0)
    return mean


# defines a function for the covariance matrix
# d determines wether we want the diagonal matrix or not

def cov(m, diagonal=True):
    c = np.dot((m - m.mean(axis=0)).T, (m - m.mean(axis=0))) / (len(m) - 1)
    if diagonal:
        c = np.diag(np.diag(c))
    return c


# makes predictions given pdfs
def predict(dist, x):
    vowel_probabilities = np.zeros((len(vowels), len(x)))

    for i in range(len(vowels)):
        vowel_probabilities[i] = dist[vowels[i]](x)
    return np.argmax(vowel_probabilities, axis=0)


# defines a function for the error rate in a given matrix
def error_rate(N, m):
    e = np.sum(m) - np.trace(m)
    print("#feil:", e)
    return e / N * 100


# prints confusion matrix and error rate
def confuse(pred, corr, label=''):
    conf = np.zeros((12, 12))
    for i, j in zip(pred, corr):
        conf[i, j] += 1
    print(label, 'confusion:\n', conf)

    print("%s error: %.2f%%" % (label, error_rate(pred.shape[0], conf)))


# train
distributions = dict()
for v in vowels:
    distributions[v] = multivariate_normal(mean(data_dict_train[v]), cov(data_dict_train[v], diagonal=False)).pdf

# predict
predicted_vowel_indices_test = predict(distributions, test_set)
# print confusion
confuse(predicted_vowel_indices_test, correct_test, 'full covar')
# GAUSS (=GMM1) diag cov

distributions = dict()
for v in vowels:
    distributions[v] = multivariate_normal(mean(data_dict_train[v]), cov(data_dict_train[v], diagonal=True)).pdf

predicted_vowel_indices_test = predict(distributions, test_set)
confuse(predicted_vowel_indices_test, correct_test, 'diag covar')

# GMM2

distributions = dict()
for v in vowels:
    gmm = GMM(2, 'diag')  # 2 mixtures, cov-type diagonal
    gmm.fit(data_dict_train[v])
    distributions[v] = gmm.score_samples

pred = predict(distributions, test_set)
confuse(pred, correct_test, 'GMM2')

# GMM3

distributions = dict()
for v in vowels:
    gmm = GMM(3, 'full')  # 3 mixtures, cov-type diag
    gmm.fit(data_dict_train[v])
    distributions[v] = gmm.score_samples

pred = predict(distributions, train_set)
confuse(pred, correct_test, 'GMM3')
