from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])
#요 데이터를 훈련해서 최소의 loss를 맹그러봐!!!

#2. 모델구성
model = Sequential()    #
model.add(Dense(1, input_dim=1))

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=9000)  #최적의 웨이트가 생성

#4. 평가, 예측
loss = model.evaluate(x,y)
print("로스:", loss)
result = model.predict([1,2,3,4,5,6,7])
print ("7의예측값 :", result)


# 로스: 0.3238094747066498
# 1/1 [==============================] - 0s 47ms/step
# 7의예측값 : [[6.8]]

# 7의예측값 : [[1.2952826]
#  [2.1898882]
#  [3.0844939]
#  [3.9790995]
#  [4.873705 ]
#  [5.7683105]
#  [6.662916 ]]
# 에포9000