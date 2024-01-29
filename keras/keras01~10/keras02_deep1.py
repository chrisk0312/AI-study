from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(1000))
model.add(Dense(500))
model.add(Dense(250))
model.add(Dense(127))
model.add(Dense(50))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100)

#4. 평가,예측
loss = model.evaluate(x,y)
print("로스 : ", loss)
result = model.predict([4])
print("4의 예측값 : ", result)


# 로스 :  0.0004569483280647546
# 1/1 [==============================] - 0s 64ms/step
# 4의 예측값 :  [[4.020547]]

