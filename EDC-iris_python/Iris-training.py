import numpy as np

class_1 = np.loadtxt("class_1", float, delimiter=',') #Iris-setosa
#class_2 = np.loadtxt("class_2") #Iris-versicolor
#class_3 = np.loadtxt("class_3") #Iris-virginica

N = 30

class_1_train = class_1[:30,:]  #[nedover, bortover], de første 30
class_1_test = class_1[-20:,:] #[nedover, bortover], de 20 siste

for k in range(N):


#if (class_1_test[0,:] == class_1_test[0,:]).all():

print(class_1_train)
print(class_1_test)
