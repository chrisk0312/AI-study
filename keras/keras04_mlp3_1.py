import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#pip install numpy
#1.데이터
x = np.array([range(10)])
print(x) # [[0 1 2 3 4 5 6 7 8 9]]
print(x.shape) # (1, 10)

x = np.array([range(1, 10)])
print(x) # [[1 2 3 4 5 6 7 8 9]]
print(x.shape) # (1, 9)

x= np.array([range(10), range(21, 31), range(201, 211)])
print(x) 
print(x.shape)
x=x.T
print(x)
print(x.shape)

y= np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]])

y=y.T


#2. 모델구성
model = Sequential()
model.add(Dense(11, input_dim=3))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(13))
model.add(Dense(2))

#3.
model.compile(loss='mse',optimizer='adam')
model.fit(x,y, epochs=100, batch_size =1)

#4. 평가,예측
loss = model.evaluate(x,y)
results = model.predict([[10,31,211]])
print("로스 :", loss)
print("[10,31,211]의 예측값 :", results)


# 실습
# 예측:[10,31,211]
# 로스 : 0.299748957157135
# [10,31,211]의 예측값 : [[10.064758  1.02563 ]]
