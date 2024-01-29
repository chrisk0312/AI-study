import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1-10, train 11-13, val 14-16
#1.데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train = np.array(range(1,11))
y_train = np.array(range(1,11))

x_val =np.array(range(11,14))
y_val =np.array(range(11,14))

x_test = np.array(range(14,17))
y_test = np.array(range(14,17))

#2. 모델구성
model= Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=16,
          validation_data=(x_val,y_val))

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
results = model.predict([30])
print("로스:", loss)
print("[30]의 예측값:", results)

# 로스: 1.0501993894577026
# [30]의 예측값: [[27.258095]]