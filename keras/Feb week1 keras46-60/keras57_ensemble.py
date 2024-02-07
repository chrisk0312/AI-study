import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, concatenate, Concatenate
from sklearn.metrics import r2_score
import pandas as pd
from keras.callbacks import EarlyStopping

#1. data
x1_datasets = np.array([range(100),range(301,401)]).T #삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101,201),range(411,511), 
                        range(150,250)]).T # 원유, 환율, 금시세
x3_datasets = np.array([range(100),range(301,401),
                        range(77,177),range(33,133)]).T


print(x1_datasets.shape,x2_datasets.shape) #(100, 2) (100, 3)

y1 = np.array(range(3001,3101)) #비트코인 종가
y2 = np.array(range(13001,13101)) #이더리움 종가

x1_train, x1test, x2_train, x2test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, y, train_size=0.7, random_state=123
 )
print(x1_train.shape, x2_train.shape, y_train.shape) #(70, 2) (70, 3) (70,)

#2-1
input1 = Input(shape=(2,))
dense1 = Dense(10, activation='relu', name ='bit1')(input1)
dense2 = Dense(10, activation='relu', name ='bit2')(dense1)
dense3 = Dense(10, activation='relu', name ='bit3')(dense2)
output1 = Dense(5, activation='relu', name ='bit4')(dense3)

# model1 = Model(inputs = input1, outputs = output1)
# model1.summary()

#2-2
input11 = Input(shape=(2,))
dense11 = Dense(100, activation='relu', name ='bit11')(input11)
dense12 = Dense(100, activation='relu', name ='bit12')(dense11)
dense13 = Dense(100, activation='relu', name ='bit13')(dense12)
output11 = Dense(5, activation='relu', name ='bit14')(dense13)

# model2 = Model(inputs = input11, outputs = output11)
# model2.summary()

#2-3 concantenate
merge1 = concatenate([output1, output11],name = 'mg1')
merge2 = Dense(7, name='mg2')(merge1)
merge3 = Dense(11, name ='mg3')(merge2)
last_output = Dense(1, name ='last')(merge3)

model = Model(inputs =[input1, input11], outputs = last_output)
model.summary()

#compile and train 
model.compile(loss= 'mse', optimizer='adam', metrics= ['mse'])
model.fit([x1_train, x2_train], y_train, batch_size=1, verbose=3,
          epochs=100, validation_split=0.2)


mse = model.evaluate([x1test, x2test], y_test ,batch_size=1)
print("mse:",mse)
y_predict = model.predict([x1test,x2test])
