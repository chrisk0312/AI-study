# save_best_only
# restore_best_weight
# 에 대한 고찰
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_breast_cancer    
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import datetime
dt = datetime.datetime.now()

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

r = int(np.random.uniform(1,1000))
r = 88
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=r)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
# scaler = MinMaxScaler().fit(x_train)    #최솟값을 0 최댓값을 1로 스케일링
scaler = StandardScaler().fit(x_train)  #정규분포로 바꿔줘서 스케일링
# scaler = MaxAbsScaler().fit(x_train)    #
# scaler = RobustScaler().fit(x_train)    #

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#model
model = Sequential()
model.add(Dense(32,input_dim=30,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

from keras.optimizers import Adam
LEARNING_RATE = 0.0001
model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=LEARNING_RATE),metrics=['mae'])
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss',mode='min',patience=200,verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss',mode='min',verbose=1,save_best_only=True,
                      period=20,
                      filepath=f"../_data/_save/MCP/k25/{dt.day}{dt.hour}_"+"{epoch:04d}-{loss:.4f}.hdf5")
rlr = ReduceLROnPlateau(monitor='val_loss',mode='auto',patience=10,verbose=1,factor=0.5)

hist = model.fit(x_train,y_train,epochs=200,batch_size=10,validation_split=0.3,verbose=2,callbacks=[es,mcp,rlr])
model.save("../_data/_save/keras25_3_save_model.h5")  #가중치와 모델 모두 담겨있다


# filename2 = path+f"{dt.month:02}{dt.day:02}_{dt.hour:02}{dt.minute:02}"

#evaluate & predict
print("============ 1. 기본출력 ============")
loss = model.evaluate(x_test,y_test,verbose=0)
y_predict = model.predict(x_test,verbose=0)

r2 = r2_score(y_test,y_predict)

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

print(f"{LEARNING_RATE=}\n{loss=}\n{r2=}\nRMSE: {RMSE(y_test,y_predict)}")


# mcp출력(마지막으로 갱신된 val_loss로)
# r=88
# loss=[6.302283763885498, 1.876548171043396]
# r2=0.9146131446215406
# RMSE: 2.510434951014859

# LEARNING_RATE=1.0
# loss=[0.6472805142402649, 0.4651944935321808]
# r2=-0.010099687753056363
# RMSE: 0.47680434273018074

# LEARNING_RATE=0.1
# loss=[0.6475515365600586, 0.4656159281730652]
# r2=-0.010672980565756829
# RMSE: 0.47693963122279487

# LEARNING_RATE=0.01
# loss=[0.0032985738944262266, 0.002811131067574024]
# r2=0.996551349275462
# RMSE: 0.027860101206717145

# LEARNING_RATE=0.001
# loss=[0.008271613158285618, 0.007348094135522842]
# r2=0.9933563528008845
# RMSE: 0.038668859644061346