import numpy as np
import Neuron as nn
from matplotlib import pyplot as  plt

#3 boyutlu siniflar olusturuluyor
class_A = np.random.normal(np.array([0,0,0]), 7, size = (200,3))
class_B = np.random.normal(np.array([0,40,40]), 6, size = (200,3))
class_C = np.random.normal(np.array([40,40,0]), 8, size = (200,3))

#siniflar cizdiriliyor
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
for x in range(200):
    ax1.scatter(class_A[x][0], class_A[x][1], class_A[x][2], color = 'red')
    ax1.scatter(class_B[x][0], class_B[x][1], class_B[x][2], color = 'blue')
    ax1.scatter(class_C[x][0], class_C[x][1], class_C[x][2], color = 'green')
ax1.set_title('Noktalar')

