import numpy as np
import Neuron as nn
from matplotlib import pyplot as  plt

#BILLINGS FUNCTION
def func_Billings(yk_1, yk_2, e):
    yk0 = (0.8-0.5*np.exp(-(yk_1**2)))*yk_1-(0.3+0.9*np.exp(-(yk_2**2)))*yk_2+(0.1*np.sin(np.pi*yk_1))+e
    return yk0

#DOT PRODUCT
def dot_product(list1,list2):
    temp = 0
    for x in range(len(list1)):
        temp = temp + list1[x]*list2[x]
    return temp

#MSE FONKSIYONU
def func_mse(list1):
    total = 0
    for x in range(len(list1)):
        total = total + list1[x]
    return total/len(list1) 

#INDEX ILE KARISTIRMA
def func_shuffle(inputs, outputs):
    temp = inputs.copy()
    ydtemp = outputs.copy()
    index = np.arange(np.shape(inputs)[0])
    np.random.shuffle(index)
    for x in range(np.shape(index)[0]):
        temp[x] = inputs[index[x]]
        ydtemp[x] = outputs[index[x]]
    inputs = temp
    outputs = ydtemp    
    return inputs, outputs

#BASLANGIC DEGERLERI
yk_2 = 0 #y(-2)
yk_1 = 0 #y(-1)

#y(0)......y(200) degerleri
billings_outputs = np.zeros(200)
e = 0.001

#BİLLİNGS SAYILARI VEKTOR BICIMINDE YAZILDI
for x in range(np.shape(billings_outputs)[0]):
    if x ==  0:     #y(0) = e
        billings_outputs[x] = func_Billings(yk_1, yk_2, e)
    if x  ==  1:    
        billings_outputs[x] = func_Billings(billings_outputs[x-1], yk_1, e)
    else:
        billings_outputs[x] = func_Billings(billings_outputs[x-1], billings_outputs[x-2], e)

#GIRISLER NORMALIZE EDILIYOR
normalized_values = billings_outputs.copy()
maximum = 0.0
for x in range(np.shape(normalized_values)[0]):
    if np.absolute(normalized_values[x]) > maximum:
        maximum = normalized_values[x]
maximum = np.absolute(maximum)
normalized_values = normalized_values/maximum

#EGITIM KUMESI GIRISLERI 0DAN 100E, ILGILI CIKISLARI 2DEN 101E
training_set = np.zeros((100,3))
training_yd = np.zeros(100)

for x in range(np.shape(training_set)[0]):
    training_set[x] = np.array([normalized_values[x], normalized_values[x+1], e])
    training_yd[x] = normalized_values[x+2]


#TEST KUMESI GIRISLERI 100DAN 150YE, ILGILI CIKISLARI 102DEN 151E
test_set = np.zeros((50,3))
test_yd = np.zeros(50)

for x in range(np.shape(test_set)[0]):
    test_set[x] = np.array([normalized_values[x+100], normalized_values[x+101], e])
    test_yd[x] = normalized_values[x+102]


#OGRENME HIZI
learning_rate = 0.15

#GIZLI KATMANLAR OLUSTURULUYOR
hiddenlayer1 = []
hiddenlayer1_noron_count =  10
y1 = []
hiddenlayer2 = []
hiddenlayer2_noron_count =  10
y2 = []
outputlayer = []
outputlayer_noron_count =  1
youtput = []


#NORONLAR OLUSUTURULUYOR
for x in range(hiddenlayer1_noron_count):
    hiddenlayer1.append(nn.hiddenNeuron(training_set[0], learning_rate, False))
    y1.append(hiddenlayer1[x].y)
for x in range(hiddenlayer2_noron_count):
    hiddenlayer2.append(nn.hiddenNeuron(y1, learning_rate, False))
    y2.append(hiddenlayer2[x].y)
for x in range(outputlayer_noron_count):
    outputlayer.append(nn.Neuron(y2, learning_rate, False))
    youtput.append(outputlayer[x].y)

#ANLIK HATA VE ORTALAMA KARESEL HATALAR TANIMLANIYOR
errorvec =  youtput[:]
instant_error_list = [None]*np.shape(training_yd)[0]
mse = list([0])

#KIRPILMIS AGIRLIKLAR VE GRADIENT VEKTORLERI OLUSTURULUYOR
weights1 = np.zeros(outputlayer_noron_count)
weights2 = np.zeros(hiddenlayer2_noron_count)
gradients1 = np.zeros(outputlayer_noron_count)
gradients2 = np.zeros(hiddenlayer2_noron_count)

#CIZIM ICIN HEESAPANLANAN DEGERLER TUTULUYOR
y_training_array = np.zeros((np.shape(training_yd)))

########## EGITIM ASAMASI ##########

for x in range(1000):
    print("epoch = ", x+1)
    
    for z in range(np.shape(training_set)[0]):
        
        #training_set, training_yd = func_shuffle(training_set, training_yd)
        
        #ILERI YOL HESABI
        for x in range(hiddenlayer1_noron_count):
            hiddenlayer1[x].update_input(np.append(training_set[z],[1]))
            hiddenlayer1[x].forward()
            y1[x] = hiddenlayer1[x].y.copy()

        for x in range(hiddenlayer2_noron_count):
            hiddenlayer2[x].update_input(np.append(y1,[1]))
            hiddenlayer2[x].forward()
            y2[x] = hiddenlayer2[x].y.copy()

      
        for x in range(outputlayer_noron_count):
            outputlayer[x].update_input(np.append(y2,[1]))
            outputlayer[x].forward()
            youtput[x] = outputlayer[x].y.copy()
        
        print("yd = ", training_yd[z]," y = ", youtput)
        errorvec =  training_yd[z] - youtput
        instant_error = 0.5*dot_product(errorvec, errorvec)
        instant_error_list[z] = instant_error.copy()

        
        #KIRPILMIS AGIRLIK MATRISLERI             
        for x in range(outputlayer_noron_count):
            weights1[x] = outputlayer[x].weights[0].copy()
        for x in range(hiddenlayer2_noron_count):
            weights2[x] = hiddenlayer2[x].weights[0].copy()  

            
        #GRADIENT HESABI VE GERIYE YAYILIM
        for x in range(outputlayer_noron_count):
            outputlayer[x].backward(training_yd[z])
            gradients1[x] = outputlayer[x].gradient.copy()
              
        for x in range(hiddenlayer2_noron_count):
            hiddenlayer2[x].backward(gradients1, weights1)
            gradients2[x] = hiddenlayer2[x].gradient.copy()
            
        for x in range(hiddenlayer1_noron_count):
            hiddenlayer1[x].backward(gradients2, weights2)
            
        y_training_array[z] = youtput[0]
    
    instant_mse = func_mse(instant_error_list)
    print("mse = ", instant_mse)
    mse.append(instant_mse)
    if (np.abs(mse[-1]-mse[len(mse)-2]) <= 0.00000000001) or (mse[-1] <= 0.0005):         #ITERASYONU DURDURMA KRITERI
        break

#EGITIM KUMESI VE ILGILI BILLINGS DURUMLARI CIZDIRILIYOR
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
training_plot1 = ax1.plot(training_yd,'g',label = 'Billings')
training_plot2 = ax1.plot(y_training_array,'r', label = 'Sinir Agı')
ax1.set_xlabel('k')
ax1.set_ylabel('y(k)')
ax1.set_title('Egitim Sonuclari')
ax1.legend()


########## TEST ASAMASI ##########

print("\n############ TEST ############\n")

y_test_array = np.zeros((np.shape(test_yd)))

for z in range(np.shape(test_set)[0]):
    
#ILERI YOL HESABI
    for x in range(hiddenlayer1_noron_count):
        hiddenlayer1[x].update_input(np.append(test_set[z],[1]))
        hiddenlayer1[x].forward()
        y1[x] = hiddenlayer1[x].y

    for x in range(hiddenlayer2_noron_count):
        hiddenlayer2[x].update_input(np.append(y1,[1]))
        hiddenlayer2[x].forward()
        y2[x] = hiddenlayer2[x].y
        
    for x in range(outputlayer_noron_count):
        outputlayer[x].update_input(np.append(y2,[1]))
        outputlayer[x].forward()
        youtput[x] = outputlayer[x].y    

    y_test_array[z] = youtput[0]
    print("yd = ", test_yd[z]," y = ", youtput[0])     


#TEST KUMESI VE ILGILI BILLINGS DURUMLARI CIZDIRILIYOR
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
test_plot1 = ax2.plot(test_yd,'g',label = 'Billings')
test_plot2 = ax2.plot(y_test_array,'r', label = 'Sinir Agı')
ax2.set_xlabel('k')
ax2.set_ylabel('y(k)')
ax2.set_title('Test Sonuclari')
ax2.legend()

#ORTALAMA KARESEL HATA ITERASYONA GORE CIZDIRILIYOR
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
mse_plot = ax3.plot(mse[1:],'g', label ='MSE')
ax3.set_xlabel('Iterasyon')
ax3.set_ylabel('Ortalama Karesel Hata')
ax3.set_title('Ortalama Karesel Hata')
ax3.legend()

plt.show()
