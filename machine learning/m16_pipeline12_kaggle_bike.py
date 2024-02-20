import numpy as np
import pandas as pd #판다스에 데이터는 넘파이 형태로 들어가있음.
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import time
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler


#1. 데이터
path = "C:\\_data\\kaggle\\bike\\"


train_csv = pd.read_csv(path + "train.csv", index_col=0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col= 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")
print(submission_csv)

print(train_csv.columns)
# ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#       'humidity', 'windspeed', 'casual', 'registered', 'count']

print(train_csv.info())
print(test_csv.info())


x = train_csv.drop(['casual','registered','count'], axis=1)
print(x)
y = train_csv['count']
print(y)

print(train_csv.index)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.77, shuffle = False, random_state=1266)
print(x_train.shape, x_test.shape) #(7620, 8) (3266, 8)
print(y_train.shape, y_test.shape) #(7620, ) (3266, )

#2. 모델구성

# model = LinearSVR(C=110)
model = make_pipeline(MinMaxScaler(), RandomForestRegressor())



#3. 컴파일, 훈련

start_time = time.time()                            

hist = model.fit(x_train, y_train)

end_time = time.time()

#4. 평가, 예측
loss = model.score(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
y_submit = model.predict(test_csv)



print("걸린 시간:", round(end_time - start_time, 2), "초")

submission_csv['count'] = y_submit

print(submission_csv)
accuracy_score = ((y_test, y_submit))

y_submit = (y_submit.round(0).astype(int))


#submission_csv.to_csv(path + "submission_29.csv", index= False)
print("음수갯수 :", submission_csv[submission_csv['count']<0].count())
print("model.score :", loss)
print("R2 스코어 :", r2)
print("정확도 :",accuracy_score)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))

import time as tm
ltm = tm.localtime(tm.time())
save_time = f"{ltm.tm_year}{ltm.tm_mon}{ltm.tm_mday}{ltm.tm_hour}{ltm.tm_min}{ltm.tm_sec}" 
#MSE : 23175.111328125
#R2 스코어 : 0.27044473122031987
#RMSE :  152.23374956711748
#RMSLE : 1.3152084898668681



#True
#MSE : 42507.94921875
#R2 스코어 : 0.107033535623137
#RMSE :  206.1745593621529
#RMSLE : 1.2849059435800314

#False
#MSE : 44973.4375
#R2 스코어 : 0.055240887807962324
#RMSE :  212.06941832566426
#RMSLE : 1.2467025777594747

#
#MSE : 22668.6015625
#R2 스코어 : 0.36016960386561647
#RMSE :  150.5609694781289
#RMSLE : 1.3219243059301249


# LinearSVR
# model.score : -0.5530860077357014
# R2 스코어 : -0.5530860077357014

