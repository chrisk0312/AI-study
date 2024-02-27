import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D 
from keras.utils import to_categorical
#1. 데이터
(x_train,y_train),(x_test,y_test)= mnist.load_data()
print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)  #(60000, 28, 28) (10000,)

print(x_train)
print(x_train[0])
print(y_train[0]) 
print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
#       dtype=int64))
print(pd.value_counts(y_test))

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
# print(x_train.shape, x_test.shape) #(60000, 784) (10000, 784)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


#2. 모델
model = Sequential()
model.add(Conv2D(32,input_shape=(3,3,1), activation='relu'))
model.add(Dense(8))
model.add(Dense(7))
#                   shape=(batch_size,input_dim)
model.add(Dense(6))
model.add(Dense(10, activation='softmax'))

# model.summary()

#(kernel_size*channels + bias) * filters 
# ((shape of width of filter*shape of height filter*number of filters in the previous layer+1(bias))*number of filters


#3. 컴파일,훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam',
             metrics = 'accuracy')
model.fit(x_train, y_train, batch_size=32, verbose=1, 
          epochs=100, validation_split=0.2
          ) 

#4. 평가,예측
results = model.evaluate(x_test,y_test)
print('loss', results[0])
print('acc', results[1])
