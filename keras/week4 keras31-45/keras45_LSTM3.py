import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6],
              [5,6,7], [6,7,8], [7,8,9], [8,9,10],
              [9,10,11],[10,11,12],
              [20,30,40],[30,40,50],[40,50,60]]
             )

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])

print(x.shape, y.shape) #(13, 3) (13,)
x = x.reshape(13,3,1)
print(x.shape, y.shape) #(13, 3, 1) (13,)

#2 
model= Sequential()
model.add(LSTM(units=10, input_shape =(3,1)))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))

#3
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000)

results= model.evaluate(x,y)
print('loss', results)
y_pred = np.array([50,60,70]).reshape(1,3,1)
y_pred = model. predict(y_pred)
print('[50,60,70]의 결과', y_pred)

# [50,60,70]의 결과 [[77.300964]]