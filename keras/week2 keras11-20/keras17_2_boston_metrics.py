# 09_1 카피

from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import r2_score

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

print(datasets.DESCR) #Describe #행 = Tnstances #print(datasets) dataset 안에 설명문을 다 넣어서 출력해줌

# [실습]
#train_size 0.7이상, 0.9이하
#R2 0.8 이상


import warnings
warnings.filterwarnings('ignore')


#1. 데이터

x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.20, random_state= 4041) #4041

print(x_train)          
print(y_train)
print(x_test)
print(x_test)

#2. 모델구성

model = Sequential()
model.add(Dense(2, input_dim= 13))
model.add(Dense(4))
model.add(Dense(8))
model.add(Dense(12, activation= 'relu'))
model.add(Dense(16))
model.add(Dense(20))
model.add(Dense(16))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam' , metrics= ['acc']) #mae 2.64084 r2 0.8278   mse 12.8935 r2 0.82
#local minimum(현재 최소값)/ global minimum(실제 최소값)

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor= 'val_loss', mode= 'min', #min-최소치(loss, val_loss등), max-최대치(정확도.r2 등), auto(자동으로 맞춰줌)
                   patience=20,verbose=1, restore_best_weights= True)
start_time = time.time() #현재시간이 들어감
hist = model.fit(x_train, y_train, epochs = 1500, batch_size= 30, validation_split= 0.3, callbacks=[es]) #hist.history안엔 loss, val_loss가 들어있음.
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print("로스 :", loss)
y_predict = model.predict(x_test) 
result = model.predict(x)


r2 = r2_score(y_test, y_predict)

print("걸린 시간 :", round(end_time - start_time, 2), "초") #def로 정의하지 않은 함수는 파이썬에서 기본으로 제공해주는 함수.

#dictionary = {'key'} : [value](':'이 꼭 들어있음.) / 두개이상 - list

from sklearn.metrics import r2_score, mean_squared_error
def RMSE(x_test, y_test):
    return np.sqrt(mean_squared_error(x_test, y_test))
rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)
print("R2 스코어 :", r2)
'''
print("=============hist=================")
print(hist)
print("=============hist.history=========")
print(hist.history)            
print("=============loss=================")
print(hist.history['loss']) 
print("=============val_loss=============")
print(hist.history['val_loss'])
print("==================================")
'''
#=============hist=================
#<keras.src.callbacks.History object at 0x000001B617478710> 래핑되어있다.

import matplotlib.pyplot as plt
#데이터 시각화
plt.rcParams['font.family'] ='Malgun Gothic' #한글깨짐 없애기
plt.rcParams['axes.unicode_minus'] =False    #한글깨짐 없애기
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], color='red', label= 'loss', marker= '.')
plt.plot(hist.history['val_loss'], color= 'blue', label = 'val_loss', marker= '.') #color= c
plt.legend(loc = 'upper right')
plt.title("보스턴 LOSS ")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()



#True
#RMSE : 4.638741771487795
#R2 스코어 : 0.7034283522895366
#로스 : 21.517925262451172

#False
#로스 : 21.81604766845703
#RMSE : 4.670765227624014
#R2 스코어 : 0.6993194660221262

#metrics
#로스 : 20.985191345214844
#RMSE : 4.580959776616478
#R2 스코어 : 0.7107707622617545
