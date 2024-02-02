from sklearn.datasets import load_diabetes

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape)

print(datasets.feature_names)
print(datasets.DESCR)

#R2 0.62이상

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import time

#1.데이터
x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=100)

print(x_train)
print(y_train)
print(x_test)
print(y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss="mae", optimizer='adam')
start_time= time.time()
model.fit(x_train, y_train, epochs=5000, batch_size=4,
          validation_split=0.3,
          verbose=2)
end_time=time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스: ", loss)
y_predict = model.predict(x_test)
results = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 스코어 :", r2)
print("걸린시간:", round(end_time - start_time,2), "초")

# 로스:  41.996177673339844
# R2 스코어 : 0.5066043137337483

# 로스:  44.149173736572266
# R2 스코어 : 0.47138845767452997