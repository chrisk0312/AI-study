import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#pip install numpy
#1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(13))
model.add(Dense(1))

#3.
model.compile(loss='mse',optimizer='adam')
model.fit(x,y, epochs=300, batch_size =1)

#4. 평가,예측
loss = model.evaluate(x,y)
results = model.predict([11000,7])
print("로스 :", loss)
print("[11000]의 예측값 :", results)