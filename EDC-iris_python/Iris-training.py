import numpy as np
#g_k sigmoid(x_ik)
#x_k input data, data fra klassematrise
#t_k targets, class labels... hmm
class_1 = np.loadtxt("class_1", float, delimiter=',') #Iris-setosa
class_2 = np.loadtxt("class_2") #Iris-versicolor
class_3 = np.loadtxt("class_3") #Iris-virginica

N = 30
alpha = 0.1 #step factor

class_1_train = class_1[:30,:]  #[nedover, bortover], de første 30
class_1_test = class_1[-20:,:] #[nedover, bortover], de 20 siste

#De forskjellige blomstenes karakteristikker
attributes = np.array([
	True, #petal length
	True, #petal width
	True, #sepal length
	True, #sepal width
])

#if (class_1_test[0,:] == class_1_test[0,:]).all(): sjekker at siste i test og første i train ikke er like, .all gjør at man kan sjekke likhet for arrays

print(class_1_train)
print(class_1_test)
