import numpy as np

class_1 = np.loadtxt("class_1", float, delimiter=',')
#class_2 = np.loadtxt("class_2")
#class_3 = np.loadtxt("class_3")

class_1_train = class_1[:30,:]  #[nedover, bortover], de første 30
class_1_test = class_1[-20:,:] #[nedover, bortover], de 20 siste

#if (class_1_test[0,:] == class_1_test[0,:]).all():

print(class_1_train)
print(class_1_test)
