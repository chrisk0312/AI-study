from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1
datasets= fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(20640, 8) (20640,)
print(np.unique(y,return_counts=True)) #(array([0, 1]), array([212, 357], dtype=int64))


#1. 데이터

x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.20,shuffle=True, random_state= 4041 ) #4041


#x_train의 scaling한 값의 기준에 따라 x_test의 값도 비율에 맞춰 바뀜. test에 범위 밖의 값이 들어가면 범위밖도 평가가 가능해서 더 좋음.


print(x_train)          
print(y_train)
print(x_test)
print(x_test)

from sklearn.preprocessing import StandardScaler, RobustScaler
mms = RobustScaler()
mms.fit(x_train)
x_train= mms.transform(x_train).reshape(x_train.shape[0],4,2,1)
x_test= mms.transform(x_test).reshape(x_test.shape[0],4,2,1)


#2. 모델구성

model = Sequential()
model.add(Conv2D(32,(2,2), input_shape = (4,2,1), activation='relu'))
model.add(Conv2D(16,(2,2), activation= 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(1))

# #3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 200, verbose = 2, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath='../_data/_save/MCP/keras26_01_MCP.hdf5')

model.compile(loss= 'mse', optimizer= 'adam' ) #mae 2.64084 r2 0.8278   mse 12.8935 r2 0.82
hist = model.fit(x_train, y_train, callbacks=[es,mcp], epochs= 2000, batch_size = 20, validation_split= 0.27)




#4. 평가, 예측

y_predict = model.predict(x_test)
results = model.evaluate(x_test, y_test)



from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("로스 :", results)
print("R2 스코어 :", r2)
