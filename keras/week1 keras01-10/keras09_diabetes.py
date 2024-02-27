from sklearn.datasets import load_diabetes
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(442, 10) (442,)

print(datasets.feature_names)
print(datasets.DESCR)


#1.데이터
x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=100)

print(x_train.shape) #(331, 10)
print(y_train.shape)
print(x_test.shape) #(111, 10)
print(y_test.shape)

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
model.fit(x_train, y_train, epochs=100, batch_size=8)
end_time=time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스: ", loss) #로스:  41.509483337402344
y_predict = model.predict(x_test)
results = model.predict(x)

r2 = r2_score(y_test, y_predict)
print("R2 스코어 :", r2) #R2 스코어 : 0.5135894923108276
print("걸린시간:", round(end_time - start_time,2), "초") #걸린시간: 5.67 초
