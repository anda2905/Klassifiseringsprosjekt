import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("vowdata_nohead.dat", dtype = [str] + 16* [float],  delimiter=' ')

print(data)