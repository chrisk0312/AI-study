import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#pip install numpy
#1.데이터
x = np.array([range(10)])
print(x) # [[0 1 2 3 4 5 6 7 8 9]]
print(x.shape) #(1, 10)
x=x.T
print(x)
print(x.shape) #(10, 1)

y= np.array([[1,2,3,4,5,6,7,8,9,10],
            [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
            [9,8,7,6,5,4,3,2,1,0]])

y=y.T

'''
#2. 모델구성
model = Sequential()
model.add(Dense(11, input_dim=1))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(13))
model.add(Dense(3))

#3.
model.compile(loss='mse',optimizer='adam')
model.fit(x,y, epochs=100, batch_size =1)

#4. 평가,예측
loss = model.evaluate(x,y)
results = model.predict([[10]])
print("로스 :", loss)
print("[10]의 예측값 :", results)

# 실습
# 예측:[10,31,211]
# 로스 : 2.610966998162212e-13
# [10]의 예측값 : [[11.          1.9999992  -0.99999934]]
'''