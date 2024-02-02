# 세이브 파일명에 시간 넣어 만들기

from sklearn.datasets import load_boston

from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
import time

# 현재 사이킷런 버전 1.3.0 보스턴 안됨. 그래서 삭제
#pip uninstall scikit-learn
#pip uninstall scikit-image                  
#pip uninstall scikit-learn-intelex
#pip install scikit-learn==1.1.3


#pip install scikit-learn==0.23.2 사이킷런 0.23.2버전을 새로 깐다.

datasets = load_boston()
print(datasets)
x = datasets.data
y = datasets.target
print(x)
print(x.shape) # (506, 13)
print(y.shape) # (506,)

print(datasets.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
# 'B' 'LSTAT']

print(datasets.DESCR) #Describe #행 = Tnstances

# [실습]
#train_size 0.7이상, 0.9이하
#R2 0.8 이상



#1. 데이터

x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.20,shuffle=True, random_state= 4041 ) #4041


#x_train의 scaling한 값의 기준에 따라 x_test의 값도 비율에 맞춰 바뀜. test에 범위 밖의 값이 들어가면 범위밖도 평가가 가능해서 더 좋음.


print(x_train)          
print(y_train)
print(x_test)
print(x_test)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

#mms = MinMaxScaler()
#mms = StandardScaler()
#mms = MaxAbsScaler()
mms = RobustScaler()

mms.fit(x_train)
x_train= mms.transform(x_train)
x_test= mms.transform(x_test)
# print(np.min(x_train)) # 0.0
# print(np.min(x_test)) # -0.028269883151149644
# print(np.min(x_train)) # 0.0
# print(np.max(x_test)) # 1.0000000000000002




#2. 모델구성

model = Sequential()
model.add(Dense(1, input_dim= 13))
model.add(Dense(9))
model.add(Dense(13))
model.add(Dense(9))
model.add(Dense(3))
model.add(Dense(1))

model.summary()





# #3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date= datetime.datetime.now()
# print(date) #2024-01-17 11:00:58.591406
# print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d-%H%M") #m=month, M=minutes
# print(date) #0117_1100
# print(type(date)) #<class 'str'>

path= '../_data/_save/MCP/' #경로(스트링data (문자))
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #filename= 에포4자리수-발로스는 소숫점4자리까지 표시. 예)1000-0.3333.hdf5
filepath = "".join([path, 'k25_', date, "_", filename]) #""공간에 ([])를 합쳐라.



model.compile(loss= 'mse', optimizer= 'adam' ) #mae 2.64084 r2 0.8278   mse 12.8935 r2 0.82
es = EarlyStopping(monitor = 'val_loss', 
                   mode = 'auto', 
                   patience = 10, 
                   verbose = 2, 
                   restore_best_weights= False)
mcp = ModelCheckpoint(monitor='val_loss', 
                      mode = 'auto', verbose= 1, 
                      save_best_only=False, 
                      filepath= filepath)


hist = model.fit(x_train, y_train, callbacks=[es,mcp], epochs= 200, batch_size = 20, validation_split= 0.27)

#model.save('../_data/_save/keras25_3_save_model.h5')
#model = load_model('../_data/_save/keras25_.hdf5')

#4. 평가, 예측
print("=============기본출력===============")
results = model.evaluate(x_test, y_test, verbose=0)
y_predict = model.predict(x_test, verbose=0) 
r2 = r2_score(y_test, y_predict)
print("로스 :", results)
print("R2 스코어 :", r2)


#print("걸린 시간 :", round(end_time - start_time, 2), "초") #def로 정의하지 않은 함수는 파이썬에서 기본으로 제공해주는 함수.

# print('===================================')
# print(hist.history['val_loss'])
# print('===================================')




#로스 : 13.562579154968262
#R2 스코어 : 0.8130732165577592
#test_size= 0.20, random_state= 4041
#epochs = 5000, batch_size= 20
# 노드 - 1, 9, 13, 9, 3, 1


#MinMax
#로스 : 12.815701484680176
#R2 스코어 : 0.8233670964311308

#StandardScaler
#로스 : 12.950993537902832
#R2 스코어 : 0.8215024368310921

#MaxAbsScaler
#로스 : 12.46930980682373
#R2 스코어 : 0.8281412427420023

#RobustScaler
#로스 : 12.295846939086914
#R2 스코어 : 0.8305320050569783



