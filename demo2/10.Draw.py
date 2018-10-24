import numpy as np
import matplotlib.pyplot as plt

plt.axis([0, 2.2, -2, 2])
xn=1.2

xs=[]
kn=[]
for k1 in range(0,2000,1):
    k=k1/1000.0
    for n in range(1000):

            xs.append(xn)
            kn.append(k)
            xn=k*xn*xn-1
    plt.scatter(kn, xs,s=0.05,c='black')
    xs=[]
    kn=[]
x1 = np.linspace(-2, 2, 100)
k1 = np.linspace(0.75, 0.751, 100)
k2 = np.linspace(1.25, 1.251, 100)
k3 = np.linspace(1.31, 1.311, 100)
k4 = np.linspace(1.40, 1.401, 100)
plt.scatter(k1, x1,s=0.02,c='b')
plt.scatter(k2, x1,s=0.02,c='b')
plt.scatter(k3, x1,s=0.02,c='b')
plt.scatter(k4, x1,s=0.02,c='b')
plt.show()

