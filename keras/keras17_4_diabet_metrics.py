from sklearn.datasets import load_diabetes
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import time
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

#1. 데이터
datasets = load_diabetes()
x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, shuffle= True, random_state = 442)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim = 10))
model.add(Dense(12))
model.add(Dense(24))
model.add(Dense(48))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(20))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam', metrics= 'acc')


es = EarlyStopping(monitor= 'val_loss', mode= 'min',
                   patience=50, verbose=0, restore_best_weights= True)

start_time = time.time()
hist = model.fit(x_train, y_train, epochs = 500, batch_size= 5, validation_split= 0.3, callbacks= [es])
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("로스 :", loss)
y_predict = model.predict(x_test)
result = model.predict(x)

r2 = r2_score(y_test, y_predict)
print("R2 스코어 :", r2)
print("걸린 시간 :", round(end_time - start_time, 2), "초")

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus']= False
plt.figure(figsize= (9,6))
plt.plot(hist.history['loss'], c = 'red', label = 'loss', marker = '.')
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss', marker = '.')
plt.legend(loc = 'upper right')
plt.title("당뇨병 LOSS")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()
#False
#로스 : 2513.03662109375
#R2 스코어 : 0.5959205821049586
#True
#로스 : 2686.110107421875
#R2 스코어 : 0.568091475683709
#metrics
#로스 : 2447.14013671875
#R2 스코어 : 0.6261307996541818
