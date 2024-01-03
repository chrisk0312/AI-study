import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

#[실습] 넘파이 리스트의 슬라이싱! 7:3
x_train = x[:7]
y_train = y[:7]

x_test = x[:7]
y_test = y[:7]

print(x_train)
print(y_train)
print(x_test)
print(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(13))
model.add(Dense(1))

#3.
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train, epochs=300, batch_size =1)

#4. 평가,예측
loss = model.evaluate(x_test,y_test)
results = model.predict([1])
print("로스 :", loss)
print("[1]의 예측값 :", results)