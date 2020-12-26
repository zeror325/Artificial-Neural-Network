import numpy as np

Shape1 = (0.9)*np.ones((5,10),dtype=float)
Shape1[0,0:10] = 0.1
Shape1[1,3:7] = 0.1

Shape2 = (0.9)*np.ones((5,10),dtype=float)
Shape2[3,3:7] = 0.1
Shape2[4,0:10] = 0.1


Shape3 = (0.9)*np.ones((5,10),dtype=float)
Shape3[1:4,0:3] = 0.1
Shape3[2,3:5] = 0.1

Shape4 = (0.9)*np.ones((5,10),dtype=float)
Shape4[1:4,7:10] = 0.1
Shape4[2,5:7] = 0.1


yd_Shape1 = np.array([1,0,0,0])
yd_Shape2 = np.array([0,1,0,0])
yd_Shape3 = np.array([0,0,1,0])
yd_Shape4 = np.array([0,0,0,1])