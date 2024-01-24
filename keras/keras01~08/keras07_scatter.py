import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#1.데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

#[검색] train과 test를 섞어서 7:3으로 자를수 있는 방법 
#사이킷런

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, shuffle=False) 
#(괄호 안에 들어가는 것을 파라미터라고 함)
#train_size = 0.7 ==디폴트 :0.75
#train_size = 0.3
#shuffle =false == #디폴트 : true
#random_state =123,
print(x_train) #[1 2 3 4 5 6 7]
print(y_train) #[1 2 3 4 6 5 7]
print(x_test) #[ 8  9 10]
print(y_test) #[ 8  9 10]

#2. 모델구성
model = Sequential()
model.add(Dense(4, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(13))
model.add(Dense(1))

#3 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=1)

#4. 평가,예측
loss= model.evaluate(x_test,y_test)
results = model.predict([x]) 
print("로스 :", loss) #로스 : 0.051522135734558105
print("[x]의 예측값 :", results) #[x]의 예측값 : [[1.1373942]



plt.scatter(x,y)
plt.plot(x, results, color ='red')
plt.show()
