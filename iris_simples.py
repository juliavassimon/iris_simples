import pandas as pd
import keras 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense
from keras.utils import to_categorical
from keras.utils import np_utils
base = pd.read_csv('iris.csv')
previsores = base.iloc[:,0:4]. values
classe = base.iloc[:,4].values
#transformar os atributos string em numericos
from sklearn.preprocessing import LabelEncoder #classe que vai transformar
#criar o objeto da classe
labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)

classe_dummy = to_categorical(classe)
from sklearn.model_selection import train_test_split

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)

#criar a rede neural

classificador = Sequential()
classificador.add(Dense(units=4, activation= 'relu', input_dim = 4))
#definir a segunda camada oculta
classificador.add(Dense(units = 4, activation= 'relu'))
#camada de saida
classificador.add(Dense(units = 4, activation='softmax'))
#compilar a rede neural
classificador.compile(optimizer = 'adam', loss ='categorical_crossentropy',
                      metrics='categorical_accuracy')
#fazer o treinamento 
classificador.fit(previsores_treinamento, classe_treinamento, 
                  batch_size=10, epochs=1000)
