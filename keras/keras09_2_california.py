import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

#1.데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape,y.shape) #(20640, 8) (20640,)

print(datasets.feature_names) # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)

#1 데이터
x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=100)

print(x_train.shape) #(14448, 8)
print(y_train.shape) #(14448,)
print(x_test.shape) #(6192, 8)
print(y_test.shape) #(20640, 8) (20640,)
print(x.shape,y.shape) #(20640, 8) (20640,)

#2. 모델구성
model = Sequential()
model.add(Dense(8, input_dim=8))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=64)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 :", loss) #로스 : 0.7849298715591431
y_predict = model.predict(x_test)
results = model.predict(x)

r2 = r2_score(y_test, y_predict)
print("R2 스코어 :",r2) #R2 스코어 : 0.4073567428266295
print("걸린시간:", round (end_time - start_time,2), "초") #걸린시간: 28.14 초