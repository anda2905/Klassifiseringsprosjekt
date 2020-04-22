import numpy as np
import matplotlib.pyplot as plt

# g_k sigmoid(x_ik)
# x_k input data, data fra klassematrise
# t_k targets, class labels... hmm
class_1_all = np.loadtxt("class_1", float, delimiter=',')  # Iris-setosa
class_2_all = np.loadtxt("class_2", float, delimiter=',')  # Iris-versicolor
class_3_all = np.loadtxt("class_3", float, delimiter=',')  # Iris-virginica


N_all = 50
N_train = 30
N_test = N_all-N_train
N_columns = 4
# Burde kanskje ha en for løkke på alpha slik at den tunes helt til all test dataen er riktig?
alpha = 0.01# step factor

# Correct answers for the training data
t_k_train = np.concatenate((
    np.repeat(np.array([[1], [0], [0]]), N_train, axis=1),
    np.repeat(np.array([[0], [1], [0]]), N_train, axis=1),
    np.repeat(np.array([[0], [0], [1]]), N_train, axis=1),
),
    axis=1
)

# Correct answers for the test data
t_k_test = np.concatenate((
    np.repeat(np.array([[1], [0], [0]]), N_test, axis=1),
    np.repeat(np.array([[0], [1], [0]]), N_test, axis=1),
    np.repeat(np.array([[0], [0], [1]]), N_test, axis=1),
),
    axis=1
)

W = np.random.rand(t_k_train.shape[0], N_columns + 1)  # W=classifying matrix, random initial values


def sigmoid(x):  # definerer sigmoid funksjonen (eq. 20)
    return 1 / (1 + np.exp(-x))


def predict(W, x):  # bruker sigmoid og gir oss gk
    # predictions
    return sigmoid(np.matmul(W, x))  # matmul gir oss matriseproduktet av to arrays


def MSE(guess, answer):  # mean square error (eq. 19)
    # square error
    return 1 / 2 * np.sum(np.linalg.norm(guess - answer, axis=0) ** 2)


def gradient(g, x, t):  # eq 22, g_k, x_k, t_k
    # gradient for W that maximizes MSE (-dW minimizes)
    return np.matmul((g - t) * g * (1 - g), x.T)

def error_rate(N,m):
    e = 0
    n = 0
    for i in m:
        e += i[0] + i[1] + i[2] - i[n]
        n += 1
    print("#feil:", e)
    return e/N

# De forskjellige blomstenes karakteristikker
attributes = np.array([
    True,  # petal length
    True,  # petal width
    True,  # sepal length
    True,  # sepal width
])

class_1 = class_1_all[:, attributes]
class_2 = class_2_all[:, attributes]
class_3 = class_3_all[:, attributes]

#Defining first 30 for test, last 20 for testing
class_1_train = class_1[:N_train, :]  # [nedover, bortover], de første 30
class_1_test = class_1[-N_test:, :]  # [nedover, bortover], de 20 siste
class_2_train = class_2[:N_train, :]  # [nedover, bortover], de første 30
class_2_test = class_2[-N_test:, :]  # [nedover, bortover], de 20 siste
class_3_train = class_3[:N_train, :]  # [nedover, bortover], de første 30
class_3_test = class_3[-N_test:, :]  # [nedover, bortover], de 20 siste


#Defining last 30 for training and first 20 for testing
#class_1_train = class_1[-N_train:, :]  # [nedover, bortover], de siste 30
#class_1_test = class_1[:N_test, :]  # [nedover, bortover], de først 20
#class_2_train = class_2[-N_train:, :]  # [nedover, bortover], de siste 30
#class_2_test = class_2[:N_test, :]  # [nedover, bortover], de 20 først
#class_3_train = class_3[-N_train:, :]  # [nedover, bortover], de siste 30
#class_3_test = class_3[:N_test, :]  # [nedover, bortover], de 20 første

train_set = np.concatenate(
    (class_1_train, class_2_train, class_3_train))  # Joining all training data into a single vector
train_set = np.concatenate((train_set, np.ones((train_set.shape[0], 1))), axis=1).T  # Adding bias coordinate

test_set = np.concatenate((class_1_test, class_2_test, class_3_test))  # Joining all test data into a single vector
test_set = np.concatenate((test_set, np.ones((test_set.shape[0], 1))), axis=1).T  # Adding bias coordinate

# print("Train set: ")
# print(train_set)
# print("    ")

# print("Test set: ")
# print(test_set)
# print("    ")


# print("Correct answers for train data: ")
# print(t_k_train)
# print("")

# print("Correct answers for test data: ")
# print(t_k_test)
# print("")

# training

maxiterations = 100000
MSE_treshold = 0.1 * test_set.shape[
    0]  # if data is perfectly classified MSE reaches 0. Unlikely that this treshold is met
dW_treshold = 0.02  # if dW is small it implies we are close to a local minimum (which is hopefully a global minimum)

# gradient descent
iterations = 0
while iterations < maxiterations:
    g_k = predict(W, train_set)
    error = MSE(g_k, t_k_train)
    dW = gradient(g_k, train_set, t_k_train)

    if error < MSE_treshold or np.linalg.norm(dW) < dW_treshold:
        print('<---- succes ---->')
        print('{0:^12}: {1:>5}'.format('iterations', iterations))
        # print('{0:^12}: {1:>5} ms'.format('time', int(1000 * (time.time() - starttime))))
        print('{0:^12}: {1:>5.2f}'.format('MSE', error))
        print('finished classifier:\n', W)
        break

    W -= alpha * dW

    iterations += 1

    if iterations % (maxiterations // 100) == 0:
        progress = 100 * iterations / maxiterations

if iterations == maxiterations:
    print('failure:')
    # print('{0:^12}: {1:>5} ms'.format('time', int(1000 * (time.time() - starttime))))
    print('MSE =', MSE(g_k, t_k_train))
    print('W =', W)

# Running the classifier for the training set.
g_k_train = predict(W, train_set)
predictions_train = np.argmax(g_k_train, axis=0)
answers_train = np.argmax(t_k_train, axis=0)

conf_train = np.zeros((3, 3))
#Confusion matrix for training data.
for i, j in zip(predictions_train, answers_train):
    conf_train[i, j] += 1
print('training confusion:\n', conf_train)
print("train feil: ", error_rate(N_train,conf_train))

#Runnig the classifier for the test set.
g_k_test = predict(W, test_set)
predictions_test = np.argmax(g_k_test, axis=0)
answers_test = np.argmax(t_k_test, axis=0)

conf_test = np.zeros((3, 3))
#confusion matrix for the test set
for i, j in zip(predictions_test, answers_test):
    conf_test[i, j] += 1
print('test confusion:\n', conf_test)

print("test feil: ", error_rate(N_test,conf_test))

class_1_PLength = class_1_all[:,0]
class_1_PWidth = class_1_all[:,1]
class_1_SLength = class_1_all[:,2]
class_1_SWidth = class_1_all[:,3]

class_2_PLength = class_2_all[:,0]
class_2_PWidth = class_2_all[:,1]
class_2_SLength = class_2_all[:,2]
class_2_SWidth = class_2_all[:,3]

class_3_PLength = class_3_all[:,0]
class_3_PWidth = class_3_all[:,1]
class_3_SLength = class_3_all[:,2]
class_3_SWidth = class_3_all[:,3]

plt.hist(class_1_PLength,10)
plt.hist(class_2_PLength,10)
plt.hist(class_3_PLength,10)

plt.subplot(3,4,1).hist(class_1_PLength,10)
plt.subplot(3,4,5).hist(class_2_PLength,10)
plt.subplot(3,4,9).hist(class_3_PLength,10)

plt.subplot(3,4,2).hist(class_1_PWidth,10)
plt.subplot(3,4,6).hist(class_2_PWidth,10)
plt.subplot(3,4,10).hist(class_3_PWidth,10)

plt.subplot(3,4,3).hist(class_1_SLength,10)
plt.subplot(3,4,7).hist(class_2_SLength,10)
plt.subplot(3,4,11).hist(class_3_SLength,10)

plt.subplot(3,4,4).hist(class_1_SWidth,10)
plt.subplot(3,4,8).hist(class_2_SWidth,10)
plt.subplot(3,4,12).hist(class_3_SWidth,10)


plt.show()
