import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time

#1 데이터
datasets = fetch_california_housing()
x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.15, random_state =282)

from sklearn.preprocessing import minmax_scale, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

mms = RobustScaler()

mms.fit(x_train)
x_train = mms.transform(x_train)
x_test = mms.transform(x_test)

#2. 모델구성
model = Sequential
model.add(Dense(8,input_dim=8))
model.add(Dense(16))
model.add(Dense(10, activation='relu'))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일,훈련 
model.compile(loss = 'mse', optimizer='adam', metrics='acc')
from keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min',patience=45, verbose=2, restore_best_weights=True)
mcp = ModelCheckpoint(monitor ='val_loss', mode= 'auto',verbose=1, save_best_only=True, filepath='c:/_data/_save/MCP/keras26_02_MCP.hdf5')



