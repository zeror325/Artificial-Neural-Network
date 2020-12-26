import matplotlib.pyplot as plt
import numpy as np
import Shapes as sh
import Neuron as nn

#VEKTOR BICIMINDEKI DATAYA NOISE EKLIYOR
def func_noise(image):
    temp = image.copy()
    noise = 0.2*(np.random.rand(1,50))
    temp = temp + noise
    temp = temp/temp.max()
    print(temp)
    return temp

#VEKTOR BICIMINDEKI DATAYA HATA EKLIYOR
def func_random(image,degree):
    temp = image.copy()
    for x in range(degree):
        rand_index = np.random.randint(0, 50)
        if temp[rand_index] == 0.1:
            temp[rand_index] = 0.9
        elif temp[rand_index] == 0.9:
            temp[rand_index] = 0.1
    return temp

#VEKTOR BICIMINDEKI DATAYI NOISE VE HATA EKLEYEREK BOZUYOR
def func_distort(image,degree):
    im = image.copy()
    im = func_random(im,degree)
    im = func_noise(im)
    return im

#MATRIS BICIMINDEKI DATAYI VEKTORE CEVIRIYOR
def func_vector(image_matrix):
    image_vector = image_matrix.copy()
    image_vector = image_vector.reshape(1,50)
    return image_vector[0]

#VEKTOR BICIMINDEKI DATAYI MATRISE CEVIRIYOR
def func_matrix(image_vector):
    image_matrix = image_vector.copy()
    image_matrix = image_matrix.reshape(5,10)
    return image_matrix

#COUNT SAYISI KADAR BOZULMUS VEKTOR KUMESI OLUSTURUYOR
def func_image_set(image_vector, count, degree):
    image_set = np.zeros((count,50))
    temp = image_vector.copy()
    for x in range(count):
        image_set[x] = func_distort(temp, degree)
    return image_set
    
#VERI SETI OLUSTURMA
def func_dataset(count_each, degree):    
    vector_Shape1 = func_vector(sh.Shape1).copy()
    Shape1_set = func_image_set(vector_Shape1, count_each-1, degree) 
    vector_Shape2 = func_vector(sh.Shape2).copy()
    Shape2_set = func_image_set(vector_Shape2, count_each-1, degree) 
    vector_Shape3 = func_vector(sh.Shape3).copy()
    Shape3_set = func_image_set(vector_Shape3, count_each-1, degree)   
    vector_Shape4 = func_vector(sh.Shape4).copy()
    Shape4_set = func_image_set(vector_Shape4, count_each-1, degree)

    original_inputs = np.concatenate((np.reshape(vector_Shape1,(1,50)), np.reshape(vector_Shape2,(1,50)), np.reshape(vector_Shape3,(1,50)), np.reshape(vector_Shape4,(1,50))), axis = 0)
    original_outputs = np.concatenate((np.reshape(sh.yd_Shape1,(1,4)), np.reshape(sh.yd_Shape2,(1,4)), np.reshape(sh.yd_Shape3,(1,4)), np.reshape(sh.yd_Shape4,(1,4))), axis = 0)
    print(original_outputs)
    data_set = np.concatenate((Shape1_set ,Shape2_set, Shape3_set, Shape4_set, original_inputs), axis = 0)
    yd = np.concatenate((np.tile(sh.yd_Shape1, (count_each-1,1)), np.tile(sh.yd_Shape2, (count_each-1,1)),
                         np.tile(sh.yd_Shape3, (count_each-1,1)), np.tile(sh.yd_Shape4,(count_each-1,1)), original_outputs),axis = 0)
    
    return data_set, yd

#BIAS TERIMI EKLEME
def func_bias(data_set,data_count):
    return np.concatenate((data_set, np.ones(data_count).reshape(data_count,1)),axis = 1)

#DOT PRODUCT
def dot_product(list1,list2):
    temp = 0
    for x in range(len(list1)):
        temp = temp + list1[x]*list2[x]
    return temp

#INDEX ILE KARISTIRMA
def func_shuffle(inputs, outputs):
    temp = inputs.copy()
    ydtemp = outputs.copy()
    index = np.arange(np.shape(inputs)[0])
    np.random.shuffle(index)
    for x in range(np.shape(index)[0]):
        temp[x] = inputs[index[x]].copy()
        ydtemp[x] = outputs[index[x]].copy()
    inputs = temp.copy()
    outputs = ydtemp.copy()
    return inputs, outputs 

#MSE FONKSIYONU
def func_mse(list1):
    total = 0
    for x in range(len(list1)):
        total = total + list1[x]
    return total/len(list1)   

#OGRENME HIZI, EGITIM KUMESINDE HER SEKILDEN KAC ADET OLACAGI ILE ILGILI DEGISKENLER
learning_rate = 0.8
count = 5   #EGITIM KUMESINDE HER SEKILDEN KAC ADET OLACAGI
degree = 1  #HER SEKIL TURU ICIN KAC NOKTANIN SIYAH VEYA BEYAZ OLARAK DEGISTIRILECEGININ SAYISI
data_count = count*4
training_set, yd = func_dataset(count,degree)

#GIZLI KATMANLAR OLUSTURULUYOR
hiddenlayer1 = []
hiddenlayer1_noron_count =  36
y1 = []
hiddenlayer2 = []
hiddenlayer2_noron_count =  27
y2 = []
hiddenlayer3 = []
hiddenlayer3_noron_count =  18
y3 = []
outputlayer = []
outputlayer_noron_count =  4
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
instant_error_list = [None]*data_count
mse = list([0])

#KIRPILMIS AGIRLIKLAR VE GRADIENT VEKTORLERI OLUSTURULUYOR
weights1 = np.zeros(outputlayer_noron_count)
weights2 = np.zeros(hiddenlayer3_noron_count)
weights3 = np.zeros(hiddenlayer2_noron_count)
gradients1 = np.zeros(outputlayer_noron_count)
gradients2 = np.zeros(hiddenlayer3_noron_count)
gradients3 = np.zeros(hiddenlayer2_noron_count)


########## EGITIM ASAMASI ##########

for e in range(10000):
    print("epoch = ", e+1)
    
    #HER ITERASYONDA KARISTIRMA ISLEMI YAPILIYOR
    training_set, yd = func_shuffle(training_set, yd)
    
    for z in range(np.shape(training_set)[0]):

        #ILERI YOL HESABI
        for x in range(hiddenlayer1_noron_count):
            hiddenlayer1[x].update_input(np.append(training_set[z],[1]))
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
        
        print("yd = ", yd[z])
        print("y = ", youtput)
        errorvec =  yd[z] - youtput
        #print("error vec = ", errorvec)
        instant_error = 0.5*dot_product(errorvec,errorvec)
        instant_error_list[z] = instant_error.copy()
        #print("instant error = ", instant_error)
       
        #KIRPILMIS AGIRLIK MATRISLERI             
        for x in range(outputlayer_noron_count):
            weights1[x] = outputlayer[x].weights[0]
        for x in range(hiddenlayer3_noron_count):
            weights2[x] = hiddenlayer3[x].weights[0]
        for x in range(hiddenlayer2_noron_count):
            weights3[x] = hiddenlayer2[x].weights[0]     
            
        #GRADIENT HESABI VE GERIYE YAYILIM
        for x in range(outputlayer_noron_count):
            outputlayer[x].backward(yd[z][x])
            gradients1[x] = outputlayer[x].gradient
        
        for x in range(hiddenlayer3_noron_count):
            hiddenlayer3[x].backward(gradients1, weights1)
            gradients2[x] = hiddenlayer3[x].gradient
        
        for x in range(hiddenlayer2_noron_count):
            hiddenlayer2[x].backward(gradients2, weights2)
            gradients3[x] = hiddenlayer2[x].gradient
            
        for x in range(hiddenlayer1_noron_count):
            hiddenlayer1[x].backward(gradients3, weights3)
    
    instant_mse = func_mse(instant_error_list)
    print("mse = ", instant_mse)
    mse.append(instant_mse)
    if (np.abs(mse[-1]-mse[len(mse)-2]) <= 0.00000001) or mse[-1]<= 0.175: #ITERASYONU DURDURMA KRITERI
        break


########## TEST ASAMASI ##########

print("\n############ TEST ############\n")

#TEST KUMESI OLUSTURULUYOR VE KARISTIRILIYOR
test_count = 20
test_set, test_yd = func_dataset(test_count,degree)
test_set, test_yd = func_shuffle(test_set,test_yd)

#TOPLAM HATAYI GÖRMEK ICIN DEGISKENLER
errorcount = 0
calculated_y_matrix = np.zeros((np.shape(test_yd)))

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


#ORIJINAL BOZULMAMIS SEKILLER CIZDIRILIYOR
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.imshow(func_matrix(sh.Shape1),cmap = 'gray')
ax1.set_title('Shape 1')
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.imshow(func_matrix(sh.Shape2),cmap = 'gray')
ax2.set_title('Shape 2')
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.imshow(func_matrix(sh.Shape3),cmap = 'gray')
ax3.set_title('Shape 3')
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.imshow(func_matrix(sh.Shape4),cmap = 'gray')
ax4.set_title('Shape 4')

#ORTALAMA KARESEL HATA ITERASYONA GORE CIZDIRILIYOR
fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
mse_plot = ax5.plot(mse[1:], 'g', label ='MSE')
ax5.set_xlabel('Iterasyon')
ax5.set_ylabel('Ortalama Karesel Hata')
ax5.set_title('Ortalama Karesel Hata')
ax5.legend()


plt.show()


