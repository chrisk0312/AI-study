#06.1 copy
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1.데이터
#x = np.array([1,2,3,4,5,6,7,8,9,10])
#y = np.array([1,2,3,4,6,5,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([31,32,33,34,35,36,37])
# y_train = np.array([1,2,3,4,5,6,7])


x_test = np.array([8,9,10])
# y_test = np.array([8,9,10])
y_test = np.array([38,39,40])


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(13))
model.add(Dense(1))

#3.
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train, epochs=1000, batch_size =1,
          verbose=3)
# verbose=0 : 침묵
# verbose=1 : 디폴트
# verbose=2 : 프로그래스바 삭제
# verbose= 나머지 값은 : 에포만 출력

#4. 평가,예측
loss = model.evaluate(x_test,y_test)
y_predict= model.predict([11])# insert x_test
print(y_predict)
print("로스 :", loss)
# print("[11]의 예측값 :", )
#로스 : 0.005348272621631622
# [11]의 예측값 : [[40.9038]]

