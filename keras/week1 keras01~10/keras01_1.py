import tensorflow as tf # tensorflow를 땡겨오고, tf라고 줄여서 쓴다.
print(tf.__version__)   # 2.15.0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])   #블라블라..
y = np.array([1,2,3])   

#2. 모델구성
model = Sequential()    #
model.add(Dense(1, input_dim=1))

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=12000)  #최적의 웨이트가 생성

#4. 평가, 예측
loss = model.evaluate(x,y)
print("로스:", loss)
result = model.predict([4])
print ("4의 예측값 : ", result)

# 로스: 0.0
# 1/1 [==============================] - 0s 43ms/step
# 4의 예측값 :  [[4.]]