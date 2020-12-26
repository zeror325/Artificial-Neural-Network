import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#SORUDA VERILEN DEGERLER
S1 = [[[0,-1],1],
      [[0,0],1],
      [[0,1],1],
      [[1,-1],1],
      [[1,0],1],
      [[1,1],1],
      [[-1,-1],1],
      [[-1,0],1],
      [[-1,1],1]]

S2 = [[[-3,3],-1],
      [[-3,1],-1],
      [[-3,0],-1],
      [[-3,-1],-1],
      [[-3,-3],-1],
      [[-1,3],-1],
      [[-1,-3],-1],
      [[0,3],-1],
      [[0,-3],-1],
      [[1,3],-1],
      [[1,-3],-1],
      [[3,3],-1],
      [[3,1],-1],
      [[3,0],-1],
      [[3,-1],-1],
      [[3,-3],-1],
      [[-2,3],-1],
      [[-3,2],-1],
      [[-3,-2],-1],
      [[-2,-3],-1],
      [[2,3],-1],
      [[3,2],-1],
      [[3,-2],-1],
      [[2,-3],-1]]

#TEK BIR VERI SETI HALINE GETIRILIYOR
veri_seti = np.concatenate((S1,S2),axis = 0)
print(veri_seti)
np.random.shuffle(veri_seti)
print(veri_seti)

#VERI SETI CIZDIRILIYOR
fig = plt.figure()
ax = fig.add_subplot(111)
for x in range(0, np.shape(S1)[0]):
    ax.plot(S1[x][0][0], S1[x][0][1], 'o', color = 'red')
for x in range(0, np.shape(S2)[0]):
    ax.plot(S2[x][0][0], S2[x][0][1], 'x',color = 'green')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title("Noktalar")
ax.text(0.0, 1, "1", transform=ax.transAxes, color = 'red')
ax.text(0.1, 1, "-1", transform=ax.transAxes, color = 'green')

#2 BOYUTLUDAN 3 BOYUTA GECIREN FONKSIYONLAR
def birim1(x1,x2):
    return x1
def birim2(x1,x2):
    return x2
def birim3(x1,x2):
    return x1**2+x2**2

#FONKSIYON BUTUN VERI SETINE UYGULANIYOR
l1,l2=2,33
xn = [[0 for x in range(l1)] for y in range(l2)]
print(xn)
for x in range(0,np.shape(veri_seti)[0]):
    xn[x][0] = [birim1(veri_seti[x][0][0],veri_seti[x][0][1]),birim2(veri_seti[x][0][0],veri_seti[x][0][1]),birim3(veri_seti[x][0][0],veri_seti[x][0][1])]
    xn[x][1] = veri_seti[x][1]
print(xn)


#FONKSIYON SONRASI VERI SETI CIZDIRILIYOR
fig2 = plt.figure()
ax1 = fig2.add_subplot(111, projection = '3d')
for x in range(0, np.shape(xn)[0]):
    if xn[x][1] == 1:
        ax1.scatter(int(xn[x][0][0]),int(xn[x][0][1]),int(xn[x][0][2]), c = 'r', marker = 'o', label = '1')
    if xn[x][1] == -1:
        ax1.scatter(int(xn[x][0][0]),int(xn[x][0][1]),int(xn[x][0][2]), c = 'g', marker = 'x', label = '-1')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('x3')
ax1.set_title("Yeni Noktalar(R2 -> R3)")
ax1.text2D(0.0, 1, "1", transform=ax1.transAxes, color = 'red')
ax1.text2D(0.1, 1, "-1", transform=ax1.transAxes, color = 'green')

xn_biased = xn
for x in range(np.shape(xn)[0]):
    xn_biased[x][0].append(1)
print(xn_biased)


############### EGITIM ASAMASI ################

w2 = [1,1,1,1]
c = 1      #OGRENME HIZI
sayac = 0
w2_eski = w2

while True:
    w2_eski = w2
    sayac = sayac + 1
    print(sayac,". iterasyon\n")
    for x in range(np.shape(xn_biased)[0]):
        yd = xn_biased[x][1]
        v = np.dot(w2,xn[x][0])
        if v <= 0:
            y = -1
        elif v > 0:
            y = 1;
        w2 = w2 + np.multiply((1/2*c*(yd-y)),xn_biased[x][0])
        print("w = \n",w2,"\n")
    print("w_eski - w = \n",(w2_eski - w2),"\n")
    if np.array_equal((w2_eski - w2),np.array(np.zeros(4),dtype = int)):
        break
print("w = \n",w2,"\n")


################# EGITIM SONUCU GRAFIK CIZDIRILMESI #################

fig3 = plt.figure()
ax2 = fig3.add_subplot(111, projection = '3d')
for x in range(0, np.shape(xn)[0]):
    if xn[x][1] == 1:
        ax2.scatter(int(xn[x][0][0]),int(xn[x][0][1]),int(xn[x][0][2]), c = 'r', marker = 'o')
    if xn[x][1] == -1:
        ax2.scatter(int(xn[x][0][0]),int(xn[x][0][1]),int(xn[x][0][2]), c = 'g', marker = 'x')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('x3')
ax2.set_title("Egitim Sonucu")
ax2.text2D(0.0, 1, "1", transform=ax2.transAxes, color = 'red')
ax2.text2D(0.1, 1, "-1", transform=ax2.transAxes, color = 'green')

xxx = np.arange(-5, 5, 0.5)
yyy = np.arange(-5, 5, 0.5)
xxx, yyy = np.meshgrid(xxx, yyy)
zzz = -(w2[0]*xxx + w2[1]*yyy + w2[3])/w2[2]

D = ax2.plot_surface(xxx,yyy,zzz, color = 'b', label = 'Ayristrici')
D._facecolors2d=D._facecolors3d
D._edgecolors2d=D._edgecolors3d
ax2.legend()

plt.show()
