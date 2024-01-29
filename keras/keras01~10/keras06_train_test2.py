import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

#[실습] 넘파이 리스트의 슬라이싱! 7:3
x_train = x[:7] # [0:7] == [:-3] == [0:-3]
y_train = y[:7]
'''
a = b # a라는 변수에  b 값을 넣어라
a == b # a와 b가 같다
'''
x_test = x[7:] #[7:10] == [=3:] == [-3:10]
y_test = y[7:]

print(x_train) #[1 2 3 4 5 6 7]
print(y_train) #[1 2 3 4 6 5 7]
print(x_test) #[ 8  9 10]
print(x_test) #[ 8  9 10]
'''
#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(13))
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train, epochs=300, batch_size =1)

#4. 평가,예측
loss = model.evaluate(x_test,y_test)
results = model.predict([1])
print("로스 :", loss)
print("[1]의 예측값 :", results)
'''