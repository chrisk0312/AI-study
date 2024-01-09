from sklearn.datasets import load_diabetes
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import time


#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape,y.shape)

print(datasets.feature_names)
print (datasets.DESCR)

x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.1, shuffle=False, random_state=100
)
 
print(x_train)
print(y_train) 
print(x_test) 
print(y_test) 

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim =10))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=64,
                 validation_split=0.3,
                 verbose=2)
end_time = time.time()

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print("로스:", loss)
y_predict = model.predict(x_test)
results= model.predict(x)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)

def RMSE(aaa,bbb):
    return np.sqrt(mean_squared_error(aaa,bbb))
rmse = RMSE(y_test, y_predict)
print("RMSE",rmse)

print("걸린시간:", round)
print("===========hist==========")
print(hist)
print("===========hist==========")
print(hist.history)
print("===========hist==========")
print(hist.history['loss'])
print("===========hist==========")
print(hist.history['val_loss'])
print("===========hist==========")




print("R2 스코어 :",r2)
print("걸린시간:", round (end_time - start_time,2), "초")


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(10,10))
plt.plot(hist.history['loss'],c='red',label = 'loss',marker=".")
plt.plot(hist.history['val_loss'], c='blue',label='val_loss',marker ='.' )
plt.legend(loc='upper right')
plt.title('캘리포니아 loss')
plt.xlabel('에포')
plt.ylabel('로스')
plt.grid()
plt.show()