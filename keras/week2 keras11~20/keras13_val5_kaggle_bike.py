#https://www.kaggle.com/competitions/bike-sharing-demand/data?select=train.csv

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_squared_log_error

#1 데이터

path = "c:\_data\kaggle\\bike\\"

train_csv = pd.read_csv(path + "train.csv",index_col=0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path +"sampleSubmission.csv",)
print(submission_csv)

print(train_csv.shape) #(10886, 11)
print(test_csv.shape) #(6493, 8)
print(submission_csv.shape) #(6493, 2)
print(train_csv.columns) #Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
                        #'humidity', 'windspeed', 'casual', 'registered', 'count'],
print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())

###########x와 y의 값을 분리
x= train_csv.drop(['casual', 'registered', 'count'], axis=1) 
print(x)
y = train_csv['count']
print(y)

x_train,x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=100,)
print(x_train.shape,x_test.shape) #(2177, 8) (8709, 8)
print(y_train.shape, y_test.shape) #(2177,) (8709,)

#2. 모델
model = Sequential()
model.add(Dense(32,input_dim = 8,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1, activation='relu'))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs=300, batch_size=32,
          validation_split=0.3,
          verbose=2)

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape) # (6493, 1)

print('===================================')
submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape) #(6493, 2)
submission_csv.to_csv(path+"submission_0109_03.csv", index= False)
print("MSE :", loss)

################################
print("음수갯수:", submission_csv[submission_csv['count']<0].count())
y_predict = model.predict(x_test)
r2 =r2_score(y_test,y_predict)
def RMSE(y_test, y_predict):
    return(np.sqrt(mean_squared_error(y_test,y_predict)))
rmse =RMSE(y_test, y_predict)
print("RMSE :",rmse)
print("R2 스코어:", r2)
'''
def RMSLE(y_test, y_predict):
    return(np.sqrt(mean_squared_log_error(y_test,y_predict)))
rmsle = RMSLE(y_test, y_predict)
print("RMSLE :", rmsle)
'''

#RMSE : 150.26206291550542
#R2 스코어: 0.2763171223075017

# RMSE : 150.1618524006552
# R2 스코어: 0.27728205583989307

# RMSE : 155.41259011172468
# R2 스코어: 0.22585555722090656

#RMSE : 150.55822806250129
#R2 스코어: 0.2734615663199157