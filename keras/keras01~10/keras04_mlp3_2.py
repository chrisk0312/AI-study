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
print(x.shape) # (3,10)
x=x.T
print(x)
print(x.shape)# (10,3)

y= np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
            [9,8,7,6,5,4,3,2,1,0]])
y=y.T
print(y.shape)# (10,3)

#2. 모델구성
model = Sequential()
model.add(Dense(11, input_dim=3))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(13))
model.add(Dense(3))

#3.
model.compile(loss='mse',optimizer='adam')
model.fit(x,y, epochs=300, batch_size =1)

#4. 평가,예측
loss = model.evaluate(x,y)
results = model.predict([[10,31,211]])
print("로스 :", loss)
print("[10,31,211]의 예측값 :", results)

# 실습
# 예측:[10,31,211]
# 로스 : 2.8947241048626893e-08
# [10,31,211]의 예측값 : [[10.999951    2.0000446  -0.99994016]]