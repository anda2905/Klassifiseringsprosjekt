import numpy as np
#g_k sigmoid(x_ik)
#x_k input data, data fra klassematrise
#t_k targets, class labels... hmm
class_1_all = np.loadtxt("class_1", float, delimiter=',') #Iris-setosa
class_2_all = np.loadtxt("class_2", float, delimiter=',') #Iris-versicolor
class_3_all = np.loadtxt("class_3", float, delimiter=',') #Iris-virginica

N_train = 30
N_all = 50
N_test = N_all-N_train
#Burde kanskje ha en for løkke på alpha slik at den tunes helt til all test dataen er riktig?
alpha = 0.1 #step factor

#De forskjellige blomstenes karakteristikker
attributes = np.array([
	True, #petal length
	True, #petal width
	True, #sepal length
	True, #sepal width
])

class_1 = class_1_all[:, attributes]
class_2 = class_2_all[:, attributes]
class_3 = class_3_all[:, attributes]

class_1_train = class_1[:N_train,:]  #[nedover, bortover], de første 30
class_1_test = class_1[-N_test:,:] #[nedover, bortover], de 20 siste
class_2_train = class_2[:N_train,:]  #[nedover, bortover], de første 30
class_2_test = class_2[-N_test:,:] #[nedover, bortover], de 20 siste
class_3_train = class_3[:N_train,:]  #[nedover, bortover], de første 30
class_3_test = class_3[-N_test:,:] #[nedover, bortover], de 20 siste

train_set = np.concatenate((class_1_train, class_2_train, class_3_train))  # Joining all training data into a single vector
train_set = np.concatenate((train_set, np.ones((train_set.shape[0], 1))), axis=1).T  # Adding bias coordinate


test_set = np.concatenate((class_1_test, class_2_test, class_3_test))  # Joining all test data into a single vector
test_set = np.concatenate((test_set, np.ones((test_set.shape[0], 1))), axis=1).T  # Adding bias coordinate

print("Train set: ")
print(train_set)
print("    ")

print("Test set: ")
print(test_set)
print("    ")


# Correct answers for the training data
t_k_train = np.concatenate((
    np.repeat(np.array([[1], [0], [0]]), N_train, axis=1),
    np.repeat(np.array([[0], [1], [0]]), N_train, axis=1),
    np.repeat(np.array([[0], [0], [1]]), N_train, axis=1),
),
    axis=1
)

t_k_test = np.concatenate((
    np.repeat(np.array([[1], [0], [0]]), N_test, axis=1),
    np.repeat(np.array([[0], [1], [0]]), N_test, axis=1),
    np.repeat(np.array([[0], [0], [1]]), N_test, axis=1),
),
    axis=1
)

print("Correct answers for train data: ")
print(t_k_train)
print("")

print("Correct answers for test data: ")
print(t_k_test)
print("")




#if (class_1_test[0,:] == class_1_test[0,:]).all(): sjekker at siste i test og første i train ikke er like, .all gjør at man kan sjekke likhet for arrays

