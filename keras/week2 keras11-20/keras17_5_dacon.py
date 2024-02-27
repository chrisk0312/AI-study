# https://dacon.io/competitions/open/235576/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time 
from keras.callbacks import EarlyStopping
from sklearn.metrics import  accuracy_score
#1. 데이터

path = "c:\_data\dacon\ddarung\\"
#print(path + aaa.csv) #c:\_data\dacon\ddarung\aaa.csv

train_csv = pd.read_csv(path + "train.csv", index_col=0)
# \ \\ / //  전부 가능
print(train_csv)
test_csv = pd.read_csv(path + "test.csv",index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path + "submission.csv")
print(submission_csv)

print(train_csv.shape) #(1459, 10)
print(test_csv.shape) # (715, 9)
print(submission_csv.shape) #(715, 2)
print(train_csv.columns) 
#['id', 'hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count']
print(train_csv.info())
print(test_csv.info())

print(train_csv.describe())

###### 결측치 처리 1. 제거########
#print(train_csv.isnull().sum())
print(train_csv.isna().sum()) #  위 아래 같음
train_csv = train_csv.dropna()
print(train_csv.isna().sum()) #  위 아래 같음
print(train_csv.info())
print(train_csv.shape) # (1328, 10)

test_csv = test_csv.fillna(test_csv.mean())
print(test_csv.info()) #  715 non-null


###########x와 y의 값을 분리
x= train_csv.drop(['count'], axis=1) 
print(x)
y = train_csv['count']
print(y)


x_train,x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.3,shuffle=False, random_state=100,)
print(x_train.shape,x_test.shape) #(929, 9) (399, 9)
print(y_train.shape, y_test.shape) #(929,) (399,)

#2. 모델
model = Sequential()
model.add(Dense(16, input_dim=9))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))


#3. 컴파일,훈련
model.compile(loss='mae', optimizer='adam', metrics =['acc'])
start_time =time.time()
es = EarlyStopping(monitor= 'val_loss',
                   mode = 'min',
                   patience=40,
                   verbose=2,
                   restore_best_weights = True,
                   )
hist = model.fit(x_train, y_train, epochs=300, batch_size=8,
          validation_split=0.3,
          callbacks= [es],
          verbose=2)
end_time =time.time()

#4.평가,예측
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
y_predict = model.predict(x_test)
print(y_submit)
print(y_submit.shape) #(715, 1)

r2= r2_score(y_test, y_predict)


def acc(y_test, y_predict):
    return(accuracy_score(y_test,y_predict))


def RMSE(aaa,bbb):
    return np.sqrt(mean_squared_error(aaa,bbb))
rmse = RMSE(y_test, y_predict)
print("RMSE",rmse)

print("걸린시간:", round)
print("===========hist==========")
print(hist)
print("===========hist==========")
print(hist.history)
print("===========hist==========")
print(hist.history['loss'])
print("===========hist==========")
print(hist.history['val_loss'])
print("===========hist==========")

print("R2 스코어 :",r2)
print("걸린시간:", round (end_time - start_time,2), "초")

print("===================================")
#########submission.csv 만들기(count 컬럼에 값만 제출)#############
submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape)

submission_csv.to_csv(path+"submission_0110_5.csv", index=False)

print("로스 :",loss)


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(10,10))
plt.plot(hist.history['loss'],c='red',label = 'loss',marker=".")
plt.plot(hist.history['val_loss'], c='blue',label='val_loss',marker ='.' )
plt.legend(loc='upper right')
plt.title('따릉이 loss')
plt.xlabel('에포')
plt.ylabel('로스')
plt.grid()
plt.show()

#로스 : 41.178062438964844