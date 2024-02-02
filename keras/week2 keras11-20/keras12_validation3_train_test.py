import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

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

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=10 )
x_train, x_val,  y_train, y_val =train_test_split (x_train, y_train, test_size = 0.2)
print(x_train)
print(y_train)
print(x_test)
print(y_test)
print(x_val)
print(y_val)


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
results = model.predict([20])
print("로스:", loss) 
print("[20]의 예측값:", results) 

# 로스: 0.01363091915845871
# [20]의 예측값: [[19.801743]]