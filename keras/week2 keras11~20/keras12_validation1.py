#06_1 카피
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1.데이터
#x = np.array([1,2,3,4,5,6,7,8,9,10])
#y = np.array([1,2,3,4,6,5,7,8,9,10])

x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,5])

x_val =np.array([6,7])
y_val =np.array([6,7])

x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(13))
model.add(Dense(1))

#3.
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train, epochs=300, batch_size =1,verbose=3,
          validation_data=(x_val,y_val))
#x_val: This typically represents the validation input data. 
#It should be a set of input samples that the model has not seen during the training phase. 
# This data is used to evaluate the model's performance on unseen examples and to monitor for potential overfitting.

#y_val: This represents the corresponding validation labels or targets. 
# It should contain the correct labels for the validation input data.


#4. 평가,예측
loss = model.evaluate(x_test,y_test)
results = model.predict([13])
print("로스 :", loss)
print("[7]의 예측값 :", results)
# 로스 : 1.1594920579227619e-08
# [7]의 예측값 : [[12.999806]]