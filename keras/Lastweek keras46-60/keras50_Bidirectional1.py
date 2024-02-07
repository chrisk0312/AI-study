#양방향 모델
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Bidirectional
from keras.callbacks import EarlyStopping

#1. 데이터
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9],
              ])
             
y = np.array([4,5,6,7,8,9,10])      

print(x.shape)    #(7, 3)

print(y.shape)    #(7,)

x = x.reshape(7,3,1)

print(x.shape)     #(7, 3, 1)

#2. 모델구성
model = Sequential()
#model.add(SimpleRNN(units= 1024, input_shape=(3,1), activation='relu')) # <- timesteps, features
#rnn model inputshape = 3-D tensor with shape (batch_size, timesteps, features).
model.add(Bidirectional(SimpleRNN(units=5), input_shape=(3,1))) #보조적인 친구, simplernn까지 랩핑하여 사용해야함.
# 연산을 한번 더 해서 연산량이 늘어남 (위 레이어만 두배가 됨)
#model.add(LSTM(units= 1024, input_shape=(3,1), activation='relu')) # <- timesteps, features
model.add(Dense(7,activation= 'relu'))
model.add(Dense(1))

model.summary()
# #3. 컴파일, 훈련
# es = EarlyStopping(monitor= 'loss', mode= 'auto', patience = 100, verbose= 0, restore_best_weights= True )
# model.compile(loss= 'mse', optimizer= 'adam', metrics= 'acc')
# model.fit(x,y,epochs=2500, callbacks= [es], batch_size= 1)

# #4. 평가, 훈련
# results = model.evaluate(x,y)
# print ('loss:', results)
# y_predict = np.array([8,9,10]).reshape(1,3,1)
# y_predict = model.predict(y_predict)
# print('예측값 :', y_predict)

# # loss: [1.324871501395819e-08, 0.0]
# 예측값 : [[11.000236]]# 1/1 [==============================] - 0s 89ms/step


