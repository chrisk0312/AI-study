'''
kingkeras@naver.com


cmd
nvidia-smi

y= wx+b (machine learning)
w(weight)
b(bias)
최적의 weight를 찾는게 중요
최소의 차이 loss
cost


deep learning < machine learning <artificial inteligince

tensorflow 행렬 연산하는  API(Application Programming Interface)

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
x = np.array([1,2,3])   
y = np.array([1,2,3])   

#2. 모델구성
model = Sequential()    
model.add(Dense(1, input_dim=1))

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=30000)  #최적의 웨이트가 생성

#4. 평가, 예측
loss = model.evaluate(x,y)
print("로스:", loss)
result = model.predict([4])
print ("4의 예측값 : ", result)

mse = Mean squared error(평균 제곱 오차)

loss = cost = error

순전파, 역전파

스칼라,백터, 행렬
스칼라의 모임이 백터
백터의 모임이 행렬

MLP(Multi-Layer Perceptron)란 지도학습에 사용되는 인공 신경망의 한 형태이다. MLP는 일반적으로 최소 하나 이상의 비선형 은닉 계층을 포함하며, 이러한 계층은 학습 데이터에서 복잡한 패턴을 추출하는 데 도움이 된다.
git config --global user.name "chrisk0312"

cmd -pip list
What does pip list do?
If you want to list all the Python packages installed in an environment, pip list command is what you are looking for. The command will return all the packages installed, along with their specific version and location.

predict x값
result   y값

scatter 그림 그리는 것

The student must demonstrate knowledge equivalent to having successfully completed 6 undergraduate Computing Science courses in the following four areas:

Theory
Computer Systems and Architecture
Software
Applications

'''