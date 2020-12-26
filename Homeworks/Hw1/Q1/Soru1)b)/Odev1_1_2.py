import numpy as np
from matplotlib import pyplot as plt
import siniflar as snf

#SINIFLARI OLUSTURUYORUZ
#A = np.random.randint(-10,10, size = (20,5))      #ILK SINIF  
#B = np.random.randint(-10,10, size = (20,5))      #IKINCI SINIF
#print("A = \n",A,"\n")
#print("B = \n",B,"\n")

#l1,l2=2,20
#A_Y = np.array([[[0 for x in range(l1)], 0] for y in range(l2)])
#print("A_Y = \n",A_Y,"\n")
#for x in range(0,np.shape(A)[0]):
#    A_Y[x][0] = A[x]
#    A_Y[x][1] = 1
#print("A_Y = \n",A_Y,"\n")

#B_Y = np.array([[[0 for x in range(l1)], 0] for y in range(l2)])
#for x in range(0,np.shape(B)[0]):
#    B_Y[x][0] = B[x]
#    B_Y[x][1] = -1
#print("B_Y = \n",B_Y,"\n")





##### YUKARIDAKI KODLAR 1 KERE CALISTIRILDIKTAN SONRA SINIFLAR KAYDEDILMISTIR(siniflar.py)




#BIAS TERIMININ EKLENMESI
AYB = snf.A_Y
for x in range(np.shape(AYB)[0]):
    AYB[x][0] = np.concatenate((AYB[x][0],np.array([1])), axis = 0)
print("AYB = \n",AYB,"\n")

BYB = snf.B_Y
for x in range(np.shape(BYB)[0]):
    BYB[x][0] = np.concatenate((BYB[x][0],np.array([1])), axis = 0)
print("BYB = \n",BYB,"\n")

#EGITIM SINIFI
E = np.concatenate((AYB[0:15,:],BYB[0:10,:]), axis = 0)
print("E = \n",E,"\n")

#TEST SINIFI
T = np.concatenate((AYB[15:20,:],BYB[10:20,:]), axis = 0)
print("T = \n",T,"\n")

#KARISTIRMA ISLEMI
np.random.shuffle(E)
np.random.shuffle(T)
print("E = \n",E,"\n")
print("T = \n",T,"\n")

#AGIRLIK VEKTORU
W = np.ones(6, dtype=int)
print("W = \n",W,"\n")

################# AGIRLIKLARIN HESAPLANMASI #################

C = 1      #OGRENME HIZI
sayac = 0
LIMIT = int(2000)
W_eski = W

while True:
    W_eski = W
    sayac = sayac + 1
    print(sayac,". iterasyon\n")
    for x in range(np.shape(E)[0]):
        print(x)
        temp = np.dot(W,E[x][0])
        if temp <= 0:
            Y = -1
        elif temp > 0:
            Y = 1;
        YD = E[x][1]
        W = W + (1/2*C*(YD-Y))*E[x][0]
        print("W = \n",W,"\n")
    print("W_eski - W = \n",(W_eski - W),"\n")
    if np.array_equal((W_eski - W),np.array([0,0,0,0,0,0],dtype = int)):
        break
    if sayac == LIMIT:
        break


 ################# EGITIM SONUCU GRAFIK CIZDIRILMESI #################

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
for x in range(0, np.shape(E)[0]):
    if E[x][1] == 1:
        ax.scatter(int(E[x][0][2]),int(E[x][0][3]),int(E[x][0][4]), c = 'r', marker = 'o')
    if E[x][1] == -1:
        ax.scatter(int(E[x][0][2]),int(E[x][0][3]),int(E[x][0][4]), c = 'g', marker = 'x')
ax.set_xlabel('x3')
ax.set_ylabel('x4')
ax.set_zlabel('x5')
ax.set_title("Egitim Sonuclari")
ax.text2D(0.0, 1, "1", transform=ax.transAxes, color = 'red')
ax.text2D(0.1, 1, "-1", transform=ax.transAxes, color = 'green')

xo = np.arange(-10,10,0.5)
yo = np.arange(-10,10,0.5)
xx, yy = np.meshgrid(xo, yo)
zz = -(W[2]*xx + W[3]*yy + W[5])/W[4]
EG = ax.plot_surface(xx,yy,zz, color = 'b', label = 'Ayristirici')
EG._facecolors2d=EG._facecolors3d
EG._edgecolors2d=EG._edgecolors3d
ax.legend()


################# TEST ASAMASI #################

print("TEST ASAMASI")
YTEST = np.array(np.zeros(len(T)))
for x in range(len(T)):
    temp = np.dot(W,T[x][0])
    if temp <= 0:
        Y = -1
    elif temp > 0:
        Y = 1
    YTEST[x] = Y

beklenen = np.zeros(15)
for x in range(np.shape(T)[0]):
    beklenen[x] = T[x][1]
print("BEKLENEN CIKTILAR\n",beklenen)
print("TEST SONUCU CIKAN SONUCLAR\n", YTEST,"\n")


################# TEST SONUCU GRAFIK CIZDIRILMESI #################

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection = '3d')
for x in range(0, np.shape(T)[0]):
    if T[x][1] == 1:
        ax2.scatter(int(T[x][0][2]),int(T[x][0][3]),int(T[x][0][4]), c = 'r', marker = 'o')
    if T[x][1] == -1:
        ax2.scatter(int(T[x][0][2]),int(T[x][0][3]),int(T[x][0][4]), c = 'g', marker = 'x')
ax2.set_xlabel('x3')
ax2.set_ylabel('x4')
ax2.set_zlabel('x5')
ax2.set_title("Test Sonuclari")
ax2.text2D(0.0, 1, "1", transform=ax2.transAxes, color = 'red')
ax2.text2D(0.1, 1, "-1", transform=ax2.transAxes, color = 'green')

zzt = -(W[2]*xx + W[3]*yy + W[5])/W[4]
TG = ax2.plot_surface(xx,yy,zzt, color = 'b', label = 'Ayristirici')    
TG._facecolors2d=TG._facecolors3d
TG._edgecolors2d=TG._edgecolors3d
ax2.legend()

plt.show()