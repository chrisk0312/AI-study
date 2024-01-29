from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])


#2. 모델구성
model = Sequential()    
model.add(Dense(3, input_dim=1))
model.add(Dense(500))
model.add(Dense(1000))
model.add(Dense(500))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100) 

#4. 평가, 예측
loss = model.evaluate(x,y)
print("로스:", loss)
result = model.predict([7])
print ("7의예측값 :", result)

# 로스: 0.3239397406578064
# 1/1 [==============================] - 0s 72ms/step
# 7의예측값 : [[6.821976]]