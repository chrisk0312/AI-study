#[실습]
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x= np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
             [9,8,7,6,5,4,3,2,1,0],
             ]
            )

y= np.array([1,2,3,4,5,6,7,8,9,10])
print(x)
print(x.shape, y.shape) #(3,10) (10,)
x=x.T
print(x.shape) #(10,3)

#2. 모델구성
model = Sequential()
model.add(Dense(11, input_dim=3))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(13))
model.add(Dense(1))

#3.
model.compile(loss='mse',optimizer='adam')
model.fit(x,y, epochs=100, batch_size =1)

#4. 평가,예측
loss = model.evaluate(x,y)
results = model.predict([[10,1.3,0]])
print("로스 :", loss)
print("[10,1.3,0]의 예측값 :", results)

# [실습] :로스 : 0.005410366225987673
#[10,1.3,0]의 예측값 : [[10.049958]]
#PS C:\study> 