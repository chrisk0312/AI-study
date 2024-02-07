#https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time


#1. 데이터
path = "c:\\_data\\daicon\\ddarung\\"
# print(path + "aaa.csv") 문자그대로 보여줌 c:\_data\daicon\ddarung\aaa.csv
# pandas에서 1차원- Series, 2차원이상은 DataFrame이라고 함.

train_csv = pd.read_csv(path + "train.csv", index_col=0) # \ \\ / // 다 가능( 예약어 사용할때 두개씩 사용) 인덱스컬럼은 0번째 컬럼이다라는뜻.
#print(train_csv)
test_csv = pd.read_csv(path +"test.csv", index_col=0)
#print(test_csv)
submission_csv = pd.read_csv(path + "submission.csv") 
#print(submission_csv)

#print(train_csv.shape)      # (1459, 10)
#print(test_csv.shape)       # (715, 9)
#print(submission_csv.shape) # (715, 2)

#print(train_csv.columns)    # id,컬럼명(header)는 데이터x, index일뿐
# [      'hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],

#print(train_csv.info())
#print(test_csv.info())

#print(train_csv.describe()) #함수는 뒤에 괄호를 꼭 넣어야 실행이 됨. 데이터의정보가 나옴.

############결측치 처리 1. 제거 ##########
#print(train_csv.isnull().sum())
#print(train_csv.isna().sum()) (둘다 똑같음)
train_csv = train_csv.fillna(train_csv.mean())  #결측치가 하나라도 있으면 행전체 삭제됨.
test_csv = test_csv.fillna(test_csv.mean())   # (0,mean)
#print(train_csv.isnull().sum())
#print(train_csv.info())
#print(train_csv.shape)      #(1328, 10)
#print(test_csv.info()) # 717 non-null


################# x와 y를 분리 ###########
x = train_csv.drop(['count',], axis=1)
#print(x)
y = train_csv['count']
#print(y)


print(train_csv.index)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.72,  shuffle= False, random_state= 6) #399 #1048 #6
#print(x_train.shape, x_test.shape) #(929, 9) (399, 9)
#print(y_train.shape, y_test.shape) #(929,) (399,)

# 로스 : 2656.447021484375
#R2 스코어 : 0.6342668951889647
#2. 모델구성

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

#mms = MinMaxScaler()
#mms = StandardScaler()
#mms = MaxAbsScaler()
mms = RobustScaler()

model = Sequential()
model.add(LSTM(1024, input_shape = (9,1) ))
model.add(Dropout(0.05))
model.add(Dense(512, activation= 'relu'))
model.add(Dropout(0.05))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(1))






    

#3. 컴파일, 훈련
import datetime
date= datetime.datetime.now()
# print(date) #2024-01-17 11:00:58.591406
# print(type(date)) #<class 'datetime.datetime'>
date = date.strftime("%m%d-%H%M") #m=month, M=minutes
# print(date) #0117_1100
# print(type(date)) #<class 'str'>

path= 'c:/_data/_save/MCP/_k28/' #경로(스트링data (문자))
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' #filename= 에포4자리수-발로스는 소숫점4자리까지 표시. 예)1000-0.3333.hdf5
filepath = "".join([path, 'k28_4', date, "_", filename]) #""공간에 ([])를 합쳐라.




from keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 1000, verbose = 2, restore_best_weights= True)
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose= 1, save_best_only=True, filepath=filepath)

model.compile(loss= 'mse', optimizer= 'adam' ) #mae 2.64084 r2 0.8278   mse 12.8935 r2 0.82
start_time = time.time()
hist = model.fit(x_train, y_train, callbacks=[mcp], epochs=10000, batch_size = 25, validation_split= 0.36)
end_time = time.time()



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
y_submit = model.predict(test_csv)
#print(y_submit)
#print(y_submit.shape)



#print(submission_csv.shape)
print("로스 :", loss)
print("R2 스코어 :", r2)
print("걸린 시간 :", round(end_time - start_time, 2), "초")

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

y_submit = (y_submit.round(0).astype(int)) #실수를 반올림한 정수로 나타내줌.


####### submission.csv 만들기 (count컬럼에 값만 넣어주면 됨) #####
submission_csv['count'] = y_submit
print(submission_csv)

#submission_csv.to_csv(path + "submission__45.csv", index= False)

path = "c:\\_data\\daicon\\ddarung\\"
import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
submission_csv.to_csv(path + f"submission_{save_time}{rmse:.3f}.csv", index=False)

#로스 : 3175.002197265625
#R2 스코어 : 0.5593716340440571
#RMSE :  56.347159447801296

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus']= False
# plt.figure(figsize= (9,6))
# plt.plot(hist.history['loss'], c = 'red', label = 'loss', marker = '.')
# plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss', marker = '.')
# plt.legend(loc = 'upper right')
# plt.title("따릉이 LOSS")
# plt.xlabel('epoch')
# plt.grid()
# plt.show()


#False
#로스 : 2293.54248046875
#R2 스코어 : 0.6335092329496075
#RMSE :  47.89094457607659

#True
#로스 : 2127.522216796875
#R2 스코어 : 0.6600380875767446
#RMSE :  46.12506761007933

#mms = MinMaxScaler()
#로스 : [1820.0758056640625, 1820.0758056640625]
#R2 스코어 : 0.7094709860009812

#mms = StandardScaler()
#R2 스코어 : 0.6445350921775757
#RMSE :  47.18982098487192

#mms = MaxAbsScaler()
#R2 스코어 : 0.6698447206814732
#RMSE :  45.47880938749681

#mms = RobustScaler()
#R2 스코어 : 0.6925418908853859
#RMSE :  43.887711905291766



#R2 스코어 : 0.6664152491454213
#RMSE :  45.714403597080334

# R2 스코어 : 0.6350364816315133
# RMSE :  47.81615942211361


#drop
#R2 스코어 : 0.710561916616094
#RMSE :  42.58217106385377

#로스 : 13.508710861206055
#R2 스코어 : 0.8138156700987883
#걸린 시간 : 363.63 초

#cpu
#걸린 시간 : 796.74 초


#LSTM
# R2 스코어 : 0.781233466400622
# RMSE :  37.02034026218852

