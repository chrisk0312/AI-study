from keras.models import Sequential, Model
from keras.layers import Dense, Input
import numpy as np


#data
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]
             ).T
y = np.array([1,2,3,4,5,6,7,8,9,10])
# print(x.shape)  # (10,2)
# print(y.shape)  # (10,)

# model (Sequential)
model = Sequential()
model.add(Dense(10,input_shape=(2,))) # (행 무시, 열 우선) input_dim에 열의 갯수만 맞추고 행은 무시
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(1))

model.summary()

# model (Functional)
input1 = Input(shape=(2,))
dense1 = Dense(10)(input1)
dense2 = Dense(9)(dense1)
dense3 = Dense(8)(dense2)
dense4 = Dense(7)(dense3)
output1 = Dense(1)(dense4)

model = Model(inputs=input1,outputs=output1)

model.summary()

#compile & fit
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=300,batch_size=1,verbose=2)

#evaluate & predict
loss = model.evaluate(x,y)
result = model.predict([[10,1.3]]) # 이것 또한 입력값이기에 열을 꼭 맞춰주기 ※행무시 열우선
print(f"LOSS: {loss}")
print(f"predict about [10, 1.3]: {result}")

# LOSS: 8.76464673638111e-06
# predict about [10, 1.3]: [[10.002944]]  