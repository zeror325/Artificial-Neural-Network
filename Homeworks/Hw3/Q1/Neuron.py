import numpy as np

#NORON SINIFI OLUSTURMA
class Neuron():
    def __init__(self,data,learning_rate,biased):
        self.data = data.copy()
        self.learning_rate = learning_rate
        self.dimension = np.shape(self.data)[0]      #len(list(data))
        self.biased = biased         #GELEN BILGININ DATANIN BIAS BILGISINI TUTUYOR, BIASLI DEGILSE BIAS EKLIYOR
        self.start()
        #print(self.__str__())

    def __str__(self):
        return "Dimensions: %d, Learning Rate: %f"% (self.dimension, self.learning_rate)

    def start(self):
        if self.biased == False:
            self.weights = np.sqrt(2/self.dimension)*np.random.rand(self.dimension+1)   #He baslatma kurali uygulanarak baslatildi
            self.data = np.append(self.data,[1])
        if self.biased == True:
            self.weights = np.sqrt(2/self.dimension)*np.random.rand(self.dimension)  #He baslatma kurali uygulanarak baslatildi
        self.weights_old = np.zeros(np.shape(self.weights))
        self.lin_com = 0
        self.y = 0
        self.error = 0
        self.momentum = 0.01
        self.a = 1          #BUTUN AKTIVASYON FONKSIYONLARININ HIZI 1 OLARAK SABITLENIP ONCELIKLE DIGER PARAMETRELER ANALIZ EDILMISTIR
        self.forward()
     
    #GIRISLERI GUNCELLEMEK ICIN KULLANILIYOR     
    def update_input(self,data):
        self.data = data.copy()
    
    #AKTIVASYON FONKSIYONU
    def func_act(self,a,lin_com):
        return np.tanh(a*lin_com)

    #AKTIVASYON FONKSIYONU TUREV
    def func_act_der(self,a,lin_com):
        return 1/((np.cosh(a*lin_com))**2)

    #ILERI YOL HESAPLAMALARI
    def forward(self):
        self.lin_com = np.dot(self.data, self.weights)
        self.y = self.func_act(self.a, self.lin_com)
    
    #GERIYE  YAYILIM ALGORITMASI
    def backward(self, yd):
        self.error = yd - self.y
        self.gradient = self.error*(self.func_act_der(self.a, self.lin_com))
        self.weights_new = self.weights + self.learning_rate*self.gradient*self.data + self.momentum*(self.weights-self.weights_old)
        self.weights_old = self.weights.copy()
        self.weights  = self.weights_new.copy()
        
class hiddenNeuron(Neuron):
    def backward(self,gradient,weights_):
        self.gradient = np.dot(gradient,weights_)*(self.func_act_der(self.a, self.lin_com))
        self.weights_new = self.weights + self.learning_rate*self.gradient*self.data + self.momentum*(self.weights-self.weights_old)
        self.weights_old = self.weights.copy()
        self.weights  = self.weights_new.copy()      
        
        