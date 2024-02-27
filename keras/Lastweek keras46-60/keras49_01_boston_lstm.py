#restore_best_weights
#save_best_only
#에 대한 고찰

from sklearn.datasets import load_boston

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
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

size = 5  #x데이터4개 y데이터 1개



def split_x(dataset, size): #a, timesteps만큼 자름
    aaa = []                #aaa = 
    for i in range(len(dataset) - size +1): # i = data길이 - timesteps +1
        subset = dataset[i : (i + size)]
        aaa.append(subset)                  # 반복
        
    return np.array(aaa)
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
model.add(LSTM(1, input_shape= (13,1)))
model.add(Dense(9))
model.add(Dropout(0.2)) #위쪽 레이어가 랜덤하게 0.2만큼 빠짐. 디폴트는 0, 평가/프레딕에서는 dropout 적용되지않음.
model.add(Dense(13))
model.add(Dropout(0.2))
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

path= 'c:/_data/_save/MCP/_k28/' #경로(스트링data (문자))
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #filename= 에포4자리수-발로스는 소숫점4자리까지 표시. 예)1000-0.3333.hdf5
filepath = "".join([path, 'k28_01_', date, "_", filename]) #""공간에 ([])를 합쳐라.




model.compile(loss= 'mse', optimizer= 'adam' ) #mae 2.64084 r2 0.8278   mse 12.8935 r2 0.82

es = EarlyStopping(monitor = 'val_loss', 
                   mode = 'auto', 
                   patience = 10, 
                   verbose = 2, 
                   restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', 
                      mode = 'auto', verbose= 1, 
                      save_best_only= True, 
                      filepath= filepath)

hist = model.fit(x_train, y_train, callbacks=[es,mcp], epochs= 1000, batch_size = 20, validation_split= 0.27)

#model.save('../_data/_save/keras25_3_save_model.h5')
#model = load_model('../_data/_save/keras25_.hdf5')

#4. 평가, 예측
print("=============기본출력===============")
results = model.evaluate(x_test, y_test, verbose=0)
y_predict = model.predict(x_test, verbose=0) 
r2 = r2_score(y_test, y_predict)
print("로스 :", results)
print("R2 스코어 :", r2)




print('===================================')
print(hist.history['val_loss'])
print('===================================')


#dropout
# 로스 : 14.783320426940918
# R2 스코어 : 0.7962482959372353


#LSTM
# 로스 : 22.96225929260254
# R2 스코어 : 0.6835217647220166