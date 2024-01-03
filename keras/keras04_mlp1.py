import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x= np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]
             ]
            )

y= np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape)
print(y.shape)
x =x.T
#[[1,1]],[2,1,1],[3,1,2],...[10,1.3]]
print(x.shape) 

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=2))
# 열, 컬럼, 속성, 특성, 차원=2// 같다
# (행무시,열우선)
model.add(Dense(1))
model.add(Dense(9))
model.add(Dense(5))
model.add(Dense(1))

#3.
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=2)

#4 평가, 예측
loss = model.evaluate(x,y)
results = model.predict([[10,1.3]])
print("로스 :", loss)
print("[10,1.3]의 예측값 :", results)

# [실습] :



