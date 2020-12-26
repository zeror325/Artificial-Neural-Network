import numpy as np
from matplotlib import pyplot as plt
import siniflar as snf

################# GERCEK DEGERLERIN HESAPLANMASI #################
def func_yd(x1,x2):
    return 3*x1+2*np.cos(x2)

#EGITIM SINIFI
#E_X1S = np.random.uniform(0,1,size=(50,1))
#print("E_X1S = \n",E_X1S,"\n")
#E_X2S = (np.random.uniform(0,1,size=(50,1)))*(np.pi/2)
#print("E_X2S = \n",E_X2S,"\n")
#E = np.concatenate((E_X1S,E_X2S),axis = 1)
#print("E = \n",E,"\n")

#TEST SINIFI
#T_X1S = np.random.rand(30,1)
#print("T_X1S = \n",T_X1S,"\n")
#T_X2S = (np.random.rand(30,1))*(np.pi/2)
#print("T_X2S = \n",T_X2S,"\n")
#T = np.concatenate((T_X1S,T_X2S),axis = 1)
#print("T = \n",T,"\n")




##### YUKARIDAKI KODLAR 1 KERE CALISTIRILDIKTAN SONRA SINIFLAR KAYDEDILMISTIR(siniflar.py)




#EKSTRA BIAS TERIMI
EB = np.concatenate((snf.E,np.ones(50).reshape((50,1))), axis = 1)              #BIASLI EGITIM SINIFI
TB = np.concatenate((snf.T,np.ones(30).reshape((30,1))), axis = 1)              #BIASLI TEST SINIFI
print("EB = \n",EB,"\n")
print("TB = \n",TB,"\n")

#EGITIM KUMESINE OUTPUTLAR EKLENIYOR
l1, l2,l3 = 2,50,30
E_Y = np.array([[[0 for x in range(l1)], 0] for y in range(l2)])
for a in range(np.shape(EB)[0]):
    E_Y[a][0] = EB[a]
    E_Y[a][1] = func_yd(EB[a][0],EB[a][1])
print("E_Y = \n",E_Y,"\n")

#TEST KUMESINE OUTPUTLAR EKLENIYOR
T_Y = np.array([[[0 for x in range(l1)], 0] for y in range(l3)])
for a in range(np.shape(TB)[0]):
    T_Y[a][0] = TB[a]
    T_Y[a][1] = func_yd(TB[a][0],TB[a][1])
print("T_Y = \n",T_Y,"\n")

#AGIRLIK VEKTORU
W = np.array([1,1,1])
print("W = \n",W,"\n")

#AKTIVASYON FONKSIYONU
def func_act(a,v):
    return 5*(1/(1+np.exp(-(a*v))))

#AKTIVASYON FONKSIYONU TUREV
def func_act_der(a,v):
    return 5*a*(np.exp((-a*v))/((1+np.exp((-a*v))**2)))


################# AGIRLIKLARIN HESAPLANMASI #################
c = 0.5           #OGREMME HIZI
ERR = np.ones(np.shape(snf.E)[0])
yv = np.ones(np.shape(snf.E)[0])
ydv = np.ones(np.shape(snf.E)[0])
dongu = 0
for k in range(10000):
    dongu = k+1
    print("\n\n\n\n\n\n\n\n\n dongu = ",k+1,"\n")
    for x in range(0,np.shape(E_Y)[0]):
        vv = np.dot(W,E_Y[x][0])
        print("vv = \n",vv,"\n")
        y = func_act(c,vv)
        print("y = \n",y,"\n")
        yv[x] = y
        yd = E_Y[x][1]
        ydv[x] = yd
        print("yd = \n",yd,"\n")
        ERROR = (1/2)*(yd-y)*(yd-y)/25
        print("ERROR = \n",ERROR,"\n")
        ERR[x] = ERROR
        W = W+(((yd-y)/5)*func_act_der(c,vv)*E_Y[x][0])
        print("W = \n",W,"\n")
    mean = (np.sum(np.absolute(ERR)))/(len(ERR))                  #ORTALAMA HATA
    print("error mean = ", mean,"\n")
    if mean <= 0.0005:
        break

    
################# EGITIM SONUCU GRAFIK CIZDIRILMESI #################

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title("Egitim Sonuclari")

xxx = np.reshape(snf.E_X1S,(1,len(snf.E_X1S)))
yyy = np.reshape(snf.E_X2S,(1,len(snf.E_X2S)))
xo = np.arange(0,1,0.001)
yo = np.arange(0,(np.pi)/2,0.001)
xx, yy = np.meshgrid(xo, yo)
YDV = func_yd(xx,yy)
IE = ax.plot_surface(xx,yy,YDV, color = 'g', label = 'Fonksiyon')
HE = ax.plot_trisurf(xxx[0], yyy[0],yv, color = 'b', label = 'Adaline')
IE._facecolors2d=IE._facecolors3d
IE._edgecolors2d=IE._edgecolors3d
HE._facecolors2d=HE._facecolors3d
HE._edgecolors2d=HE._edgecolors3d
ax.legend()



################# TEST #################
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n ################# TEST ################# \n")

yvt = np.ones(np.shape(snf.E)[0])
for x in range(0,np.shape(T_Y)[0]):
    vv = np.dot(W,T_Y[x][0])
    print("vv = \n",vv,"\n")
    y = func_act(c,vv)
    yvt[x] = y
    print("y = \n",y,"\n")
    yd = T_Y[x][1]   
    print("yd = \n",yd,"\n")
    ERROR = (1/2)*(yd-y)*(yd-y)/25
    print("ERROR = \n",ERROR,"\n")

    
################# TEST SONUCU GRAFIK CIZDIRILMESI #################

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection = '3d')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('y')
ax2.set_title("Test Sonuclari")

xxx2 = np.reshape(snf.T_X1S,(1,len(snf.T_X1S)))
yyy2 = np.reshape(snf.T_X2S,(1,len(snf.T_X2S)))

IT = ax2.plot_surface(xx,yy,YDV, color = 'g', label = 'Fonksiyon')
print(np.shape(xx))
print(xxx)
HT = ax2.plot_trisurf(xxx2[0], yyy2[0],yvt, color = 'r', label = 'Adaline')
IT._facecolors2d=IT._facecolors3d
IT._edgecolors2d=IT._edgecolors3d
HT._facecolors2d=HT._facecolors3d
HT._edgecolors2d=HT._edgecolors3d
ax2.legend()

print(dongu)

plt.show()
