# https://dacon.io/competitions/open/235576/leaderboard

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

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


x_train,x_test, y_train, y_test = train_test_split(x,y, train_size=0.2, random_state=100,)
print(x_train.shape,x_test.shape) #(929, 9) (399, 9)
print(y_train.shape, y_test.shape) #(929,) (399,)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
scaler = MinMaxScaler()
scaler = RobustScaler()


scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))
print(np.min(x_test))
print(np.max(x_train))
print(np.max(x_test))

#2. 모델
model = Sequential()
model.add(Dense(16, input_dim=9))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))


#3. 컴파일,훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_split=0.3,
          verbose=2)

#4.평가,예측
loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape) #(715, 1)


print("===================================")
#########submission.csv 만들기(count 컬럼에 값만 제출)#############
submission_csv['count'] = y_submit
print(submission_csv)
print(submission_csv.shape)

submission_csv.to_csv(path+"submission_0109_2.csv", index=False)

print("로스 :",loss) #로스 : 47.53935623168945
# 로스 : 49.320796966552734