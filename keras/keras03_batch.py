#from tensorflow.keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import keras
print("tf 버전 :", tf.__version__)
print("keras 버전 :", keras.__version__)

#1. 데이터
x = np.array([1,2,3,4,5,6,])
y = np.array([1,2,3,5,4,6,])

#2. 모델구성
model = Sequential()
model.add(Dense(1,input_dim=1))
model.add(Dense(500))
model.add(Dense(1000))
model.add(Dense(500))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=5)

#4 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([7])
print("로스 :", loss)
print("7의 예측값 :", results)

# 로스 : 0.3272707462310791
# 7의예측값 : [[6.7545266]]