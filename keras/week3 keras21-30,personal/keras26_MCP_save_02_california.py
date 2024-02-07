#copy 17-3

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time
#1.데이터
datasets = fetch_california_housing()
x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.15, random_state= 282)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

#mms = MinMaxScaler()
#mms = StandardScaler()
#mms = MaxAbsScaler()
mms = RobustScaler()

mms.fit(x_train)
x_train= mms.transform(x_train)
x_test= mms.transform(x_test)

#2. 모델구성

model = Sequential()
model.add(Dense(8, input_dim = 8))
model.add(Dense(16))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam', metrics= 'acc')

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 45, verbose = 2, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath='../_data/_save/MCP/keras26_02_MCP.hdf5')

model.compile(loss= 'mse', optimizer= 'adam' ) #mae 2.64084 r2 0.8278   mse 12.8935 r2 0.82
hist = model.fit(x_train, y_train, callbacks=[es,mcp], epochs= 700, batch_size = 200, validation_split= 0.27)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)
result = model.predict(x)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("로스 :", loss)
print("R2 스코어 :", r2)
#print("걸린 시간 :", round(end_time - start_time, 2), "초" )

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus']= False
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], color= 'red', label= 'loss', marker= '.')
plt.plot(hist.history['val_loss'], color= 'blue', label = 'val_loss', marker= '.')
plt.legend(loc = 'upper right')
plt.title("캘리포니아 loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.show()

#False
# 로스 : 0.637042224407196
# R2 스코어 : 0.5180010905343362

#True
#로스 : 0.6241012811660767
#R2 스코어 : 0.5277923043615612

#metrics
#로스 : 0.4702024757862091
#R2 스코어 : 0.6442353914257194

#MinMax
#로스 : [0.3675292730331421, 0.0019379844889044762]
#R2 스코어 : 0.7219199654372063

#StandardScaler()
#로스 : [0.3095373809337616
#R2 스코어 : 0.7657978579939255

# MaxAbsScaler
#로스 : [0.36424240469932556, 0.002583979396149516]
#R2 스코어 : 0.7244069184233091

#RobustScaler
#로스 : [0.34082022309303284, 0.002583979396149516]
#R2 스코어 : 0.7421286277994633