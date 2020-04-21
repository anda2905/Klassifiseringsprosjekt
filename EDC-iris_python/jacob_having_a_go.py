import numpy as np
import matplotlib.pyplot as plt
import time

x1all = np.loadtxt('class_1', delimiter=',')
x2all = np.loadtxt('class_2', delimiter=',')
x3all = np.loadtxt('class_3', delimiter=',')

# attributes to classify by
attributes = np.array([
    False,  # petal length
    False,  # petal width
    True,  # sepal length
    True,  # sepal width
])

x1 = x1all[:, attributes]
x2 = x2all[:, attributes]
x3 = x3all[:, attributes]

Ntot, dimx = x1.shape
# choose amount of samples used for traing, rest is used for testing
Ndtot = 30
Nttot = Ntot - Ndtot

x1d = x1[:Ndtot]
x1t = x1[Ndtot:]
x2d = x2[:Ndtot];
x2t = x2[Ndtot:]
x3d = x3[:Ndtot];
x3t = x3[Ndtot:]

x_train = np.concatenate((x1d, x2d, x3d))  # Join training data into single vector
x_train = np.concatenate((x_train, np.ones((x_train.shape[0], 1))), axis=1).T  # adding bias coordinate

# answer sheat for training data
y_train = np.concatenate((
    np.repeat(np.array([[1], [0], [0]]), Ndtot, axis=1),
    np.repeat(np.array([[0], [1], [0]]), Ndtot, axis=1),
    np.repeat(np.array([[0], [0], [1]]), Ndtot, axis=1),
),
    axis=1
)

x_test = np.concatenate((x1t, x2t, x3t))  # Join test data into single vector
x_test = np.concatenate((x_test, np.ones((x_test.shape[0], 1))), axis=1).T  # adding bias coordinate

# answer sheat for test data
y_test = np.concatenate((
    np.repeat(np.array([[1], [0], [0]]), Nttot, axis=1),
    np.repeat(np.array([[0], [1], [0]]), Nttot, axis=1),
    np.repeat(np.array([[0], [0], [1]]), Nttot, axis=1),
),
    axis=1
)

W = np.random.rand(y_train.shape[0], dimx + 1)  # W=classifying matrix, random initial values


def sigmoid(x): #eq 20
    return 1 / (1 + np.exp(-x))


def predict(W, x): #gir oss gk
    # predictions
    return sigmoid(np.matmul(W, x))


def MSE(guess, answer): #formel 19
    # square error
    return 1 / 2 * np.sum(np.linalg.norm(guess - answer, axis=0) ** 2)


def gradient(g, x, y): #eq 22
    # gardient for W that maximizes MSE (-dW minimizes)
    return np.matmul((g - y) * g * (1 - g), x.T)


# training
stepsize = 0.01  # too big = no convergence, too small = takes crazy long time
maxiterations = 100000
MSE_treshold = 0.1 * x_train.shape[
    0]  # if data is perfectly classified MSE reaches 0. Unlikely that this treshold is met
dW_treshold = 0.02  # if dW is small it implies we are close to a local minimum (which is hopefully a global minimum)

# timing
starttime = time.time()

# gradient descent
iterations = 0
while iterations < maxiterations:
    g = predict(W, x_train)
    error = MSE(g, y_train)
    dW = gradient(g, x_train, y_train)

    if error < MSE_treshold or np.linalg.norm(dW) < dW_treshold:
        print('<---- succes ---->')
        print('{0:^12}: {1:>5}'.format('iterations', iterations))
        print('{0:^12}: {1:>5} ms'.format('time', int(1000 * (time.time() - starttime))))
        print('{0:^12}: {1:>5.2f}'.format('MSE', error))
        print('finished classifier:\n', W)
        break

    W -= stepsize * dW

    iterations += 1

    if iterations % (maxiterations // 100) == 0:
        progress = 100 * iterations / maxiterations
        print(' {0:02.0f}% time passed: {1:>6} ms'.format(progress, int(1000 * (time.time() - starttime))))

if iterations == maxiterations:
    print('failure:')
    print('{0:^12}: {1:>5} ms'.format('time', int(1000 * (time.time() - starttime))))
    print('MSE =', MSE(g, y_train))
    print('W =', W)

# confusion matrix training data
g = predict(W, x_train)
predictions = np.argmax(g, axis=0)
answers = np.argmax(y_train, axis=0)

conf = np.zeros((3, 3));

for i, j in zip(predictions, answers):
    conf[i, j] += 1
print('training confusion:\n', conf)

# confusion matrix test data
g = predict(W, x_test)
predictions = np.argmax(g, axis=0)
answers = np.argmax(y_test, axis=0)

conf = np.zeros((3, 3));

for i, j in zip(predictions, answers):
    conf[i, j] += 1
print('test confusion:\n', conf)

# plotting the predictions

predicted_0 = x_test[:, predictions == 0].T
predicted_1 = x_test[:, predictions == 1].T
predicted_2 = x_test[:, predictions == 2].T
print("Predicted: ")
print(predictions)


solution_0 = x_test[:, answers == 0].T
solution_1 = x_test[:, answers == 1].T
solution_2 = x_test[:, answers == 2].T

print("solution: ")
print(answers)

wrong_class = x_test[:, predictions != answers].T

cols = 2;
rows = np.count_nonzero(attributes);
for i in range(cols * rows // 2):
    at1 = i
    at2 = (i + 1) % np.count_nonzero(attributes)

    plt.subplot(rows, cols, 2 * i + 1).set_title('true data (' + str(at1) + ', ' + str(at2) + ')')
    plt.plot(solution_0[:, at1], solution_0[:, at2], 'ro')
    plt.plot(solution_1[:, at1], solution_1[:, at2], 'go')
    plt.plot(solution_2[:, at1], solution_2[:, at2], 'bo')

    plt.subplot(rows, cols, 2 * i + 2).set_title('predicted data (' + str(at1) + ', ' + str(at2) + ')')
    plt.plot(predicted_0[:, at1], predicted_0[:, at2], 'ro')
    plt.plot(predicted_1[:, at1], predicted_1[:, at2], 'go')
    plt.plot(predicted_2[:, at1], predicted_2[:, at2], 'bo')

    plt.plot(wrong_class[:, at1], wrong_class[:, at2], color='black', marker='x', linestyle='')

plt.show()
