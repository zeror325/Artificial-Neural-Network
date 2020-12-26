import numpy as np
import Neuron as nn
from matplotlib import pyplot as  plt

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

#DATA SETI OLUSTURMA FONKSIYONU
def data_set(inputs,outputs,start,stop):
    input_set = inputs[start-1:stop,0:np.shape(inputs)[1]].copy()
    output_set = outputs[start-1:stop,0:np.shape(outputs)[1]].copy()
    return input_set,output_set

#BIAS TERIMI EKLEME
def func_bias(data_set,data_count):
    return np.concatenate((data_set, np.ones(data_count).reshape(data_count,1)),axis = 1)

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

#iris.data DOSYASINDA DATALAR STRING OLARAK OKUNUP LIST'E AKTARILIYOR
alldata = list()
myfile = open("iris.data","r")
alldata = myfile.read()
alldata = alldata.split("\n")
for x in range(len(alldata)):
    alldata[x] = alldata[x].split(",")


#ISIM KISMI CIKARILIP NUMPY ARRAY'E CEVIRILIYOR
all_data_matrix = np.zeros((150,4))
for x in range(len(alldata)):
    for y in range(len(alldata[0])-1):
        all_data_matrix[x][y] = float(alldata[x][y])
print(all_data_matrix)

#GIRISLER NORMALIZE EDILIYOR
normalized_data = np.zeros((150,4))
maximum = 0.0
for x in range(np.shape(all_data_matrix)[0]):
    for y in range(np.shape(all_data_matrix)[1]):
       if all_data_matrix[x][y] > maximum:
           maximum = all_data_matrix[x][y].copy()
print(maximum)       
normalized_data = (all_data_matrix/maximum)
print(normalized_data)


#ILGILI LABELLAR setosa ICIN [0, 0, 1]  versicolor ICIN [0,1,0] virginica ICIN [1,0,0] VE OUTPUT MATRISI OLUSTURULUYOR
setosa_yd = np.array([0,0,1])
versicolor_yd = np.array([0,1,0])
virginica_yd = np.array([1,0,0])
all_yd_matrix = np.concatenate((np.tile(setosa_yd, (50,1)), np.tile(versicolor_yd, (50,1)), np.tile(virginica_yd, (50,1))), axis=0)


#BUTUN DATA ONCE KARISTIRILIYOR SONRASINDA EGITIM VE TEST ICIN 2YE AYRILIYOR)
normalized_data, all_yd_matrix = func_shuffle(normalized_data, all_yd_matrix)

#EGITIM KUMESI VE  TEST KUMESI(ILK 100 VERI EGITIM KUMESI SONRAKI 50 VERI TEST KUMESI)
training_data_count = 100
training_set, training_yd = data_set(normalized_data, all_yd_matrix, 1,training_data_count)
test_set, test_yd = data_set(normalized_data, all_yd_matrix, training_data_count,150)


#OGRENME HIZI
learning_rate = 0.15

#GIZLI KATMANLAR OLUSTURULUYOR
hiddenlayer1 = []
hiddenlayer1_noron_count =  5
y1 = []
hiddenlayer2 = []
hiddenlayer2_noron_count =  5
y2 = []
hiddenlayer3 = []
hiddenlayer3_noron_count =  5
y3 = []
outputlayer = []
outputlayer_noron_count =  3
youtput = []


#NORONLAR OLUSUTURULUYOR
for x in range(hiddenlayer1_noron_count):
    hiddenlayer1.append(nn.hiddenNeuron(training_set[0], learning_rate, False))
    y1.append(hiddenlayer1[x].y)
for x in range(hiddenlayer2_noron_count):
    hiddenlayer2.append(nn.hiddenNeuron(y1, learning_rate, False))
    y2.append(hiddenlayer2[x].y)
for x in range(hiddenlayer3_noron_count):
    hiddenlayer3.append(nn.hiddenNeuron(y2, learning_rate, False))
    y3.append(hiddenlayer3[x].y)   
for x in range(outputlayer_noron_count):
    outputlayer.append(nn.Neuron(y3, learning_rate, False))
    youtput.append(outputlayer[x].y)

#ANLIK HATA VE ORTALAMA KARESEL HATALAR TANIMLANIYOR
errorvec =  youtput[:]
instant_error_list = [None]*training_data_count
mse = list([0])

#KIRPILMIS AGIRLIKLAR VE GRADIENT VEKTORLERI OLUSTURULUYOR
weights1 = np.zeros(outputlayer_noron_count)
weights2 = np.zeros(hiddenlayer3_noron_count)
weights3 = np.zeros(hiddenlayer2_noron_count)
gradients1 = np.zeros(outputlayer_noron_count)
gradients2 = np.zeros(hiddenlayer3_noron_count)
gradients3 = np.zeros(hiddenlayer2_noron_count)


########## EGITIM ASAMASI ##########

for x in range(1000):
    print("epoch = ", x+1)
    
    #HER ITERASYONDA KARISTIRMA ISLEMI YAPILIYOR
    training_set, training_yd = func_shuffle(training_set, training_yd)
    
    for z in range(np.shape(training_set)[0]):
                
        #ILERI YOL HESABI
        for x in range(hiddenlayer1_noron_count):
            hiddenlayer1[x].update_input(np.append(training_set[z],[1]))
            hiddenlayer1[x].forward()
            y1[x] = hiddenlayer1[x].y.copy()

        for x in range(hiddenlayer2_noron_count):
            hiddenlayer2[x].update_input(np.append(y1,[1]))
            hiddenlayer2[x].forward()
            y2[x] = hiddenlayer2[x].y.copy()

        for x in range(hiddenlayer3_noron_count):
            hiddenlayer3[x].update_input(np.append(y2,[1]))
            hiddenlayer3[x].forward()
            y3[x] = hiddenlayer3[x].y.copy()
        
        for x in range(outputlayer_noron_count):
            outputlayer[x].update_input(np.append(y3,[1]))
            outputlayer[x].forward()
            youtput[x] = outputlayer[x].y.copy()
        
        print("yd = ", training_yd[z]," y = ", youtput)
        errorvec =  training_yd[z] - youtput
        instant_error = 0.5*dot_product(errorvec, errorvec)
        instant_error_list[z] = instant_error.copy()
        #print("instant error = ", instant_error)
       
        #KIRPILMIS AGIRLIK MATRISLERI             
        for x in range(outputlayer_noron_count):
            weights1[x] = outputlayer[x].weights[0].copy()
        for x in range(hiddenlayer3_noron_count):
            weights2[x] = hiddenlayer3[x].weights[0].copy()
        for x in range(hiddenlayer2_noron_count):
            weights3[x] = hiddenlayer2[x].weights[0].copy()  
            
        #GRADIENT HESABI VE GERIYE YAYILIM
        for x in range(outputlayer_noron_count):
            outputlayer[x].backward(training_yd[z][x])
            gradients1[x] = outputlayer[x].gradient.copy()
        
        for x in range(hiddenlayer3_noron_count):
            hiddenlayer3[x].backward(gradients1, weights1)
            gradients2[x] = hiddenlayer3[x].gradient.copy()
        
        for x in range(hiddenlayer2_noron_count):
            hiddenlayer2[x].backward(gradients2, weights2)
            gradients3[x] = hiddenlayer2[x].gradient.copy()
            
        for x in range(hiddenlayer1_noron_count):
            hiddenlayer1[x].backward(gradients3, weights3)
    
    instant_mse = func_mse(instant_error_list)
    print("mse = ", instant_mse)
    mse.append(instant_mse)
    if np.abs(mse[-1]-mse[len(mse)-2]) <= 0.00000001 or mse[-1]<= 0.01:        #ITERASYONU DURDURMA KRITERI
        break
    
    

########## TEST ASAMASI ##########

print("\n############ TEST ############\n")

#TOPLAM HATAYI GÖRMEK ICIN DEGISKENLER
calculated_y_matrix = np.zeros((np.shape(test_yd)))
errorcount = 0

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

    for x in range(hiddenlayer3_noron_count):
        hiddenlayer3[x].update_input(np.append(y2,[1]))
        hiddenlayer3[x].forward()
        y3[x] = hiddenlayer3[x].y
        
    for x in range(outputlayer_noron_count):
        outputlayer[x].update_input(np.append(y3,[1]))
        outputlayer[x].forward()
        youtput[x] = outputlayer[x].y    

    print(youtput)
    maximuminy = max(youtput)
    for x in range(len(youtput)):
        if youtput[x] == maximuminy:
            youtput[x] = 1
        else:
            youtput[x] = 0
    array_youtput = np.array(youtput)
    if not(np.array_equal(array_youtput,test_yd[z])):
        errorcount = errorcount + 1
    calculated_y_matrix[z] = array_youtput
    print("yd = ", test_yd[z]," y = ", youtput)     

#DOGRULUK HESABI
accuracy = (np.shape(test_yd)[0] - errorcount)/ np.shape(test_yd)[0] * 100
print("Accuracy = ", accuracy)

#ORTALAMA KARESEL HATA ITERASYONA GORE CIZDIRILIYOR
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
mse_plot = ax3.plot(mse[1:],'g', label ='MSE')
ax3.set_xlabel('Iterasyon')
ax3.set_ylabel('Ortalama Karesel Hata')
ax3.set_title('Ortalama Karesel Hata')
ax3.legend()





        