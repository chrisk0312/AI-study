from sklearn.datasets import load_boston
from keras.models import Sequential, load_model,Model
from keras.layers import Dense, Dropout
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
import time

datasets =  load_boston()
print(datasets)
x= datasets.data
y= datasets.target
print(x)
print(x.shape)
print(y.shape)
print(datasets.feature_names)
print(datasets.DESCR)

x= np.array(datasets.data)
y= np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,shuffle=True, random_state=100)
print(x_train)
print(y_train)
print(x_test)
print(y_test)

from sklearn.preprocessing import StandardScaler
mms = StandardScaler

mms.fit(x_train)
x_train = mms.transform(x_train)
x_test =  mms.transform(x_test)

#2 모델구성(함수형)
input1 =  input(shape=(2,))
dense1 = Dense(10)(input1)
dense2 = Dense(9)(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(8, activation='relu')(drop1)
dense4 = Dense(7)(dense3) 
output1 = Dense(1)(dense4)
model = Model(inputs = input1, outputs =output1)

model.summary()

#3.컴파일,훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
print(date)
print(type(date))
date = date.strftime("%m%d_%H%M")
print(date)
print (type(date))

path = 'c:/_data/save/MCP'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = "".join([path, 'k28_01_', date,'_', filename])
#'../_data/_save/MCP/k25_0117_1058_0101-0.3333.hdf5'

es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 10, verbose = 2, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath=filepath)

model.compile(loss= 'mse', optimizer= 'adam',metrics= ['acc'] ) #mae 2.64084 r2 0.8278   mse 12.8935 r2 0.82
hist = model.fit(x_train, y_train,epochs= 100, batch_size = 32, validation_split= 0.27,callbacks=[es,mcp],)


# model = load_model('../_data/_save/MCP/keras25_MCP1.hdf5')

#4. 평가, 예측
print("=========================1.load_model=======================")
results = model.evaluate(x_test, y_test, verbose=0)
print("로스 :", results)

y_predict = model.predict(x_test) 
# result = model.predict(x,verbose=0)

r2 = r2_score(y_test, y_predict)
print("R2 스코어 :", r2)

print('==============================')
# print(hist.history['val_loss'])
print('==============================')