import numpy as np
import pandas as pd #판다스에 데이터는 넘파이 형태로 들어가있음.
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import time
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

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


#columns = x.feature_names
columns = x.columns
x = pd.DataFrame(x,columns=columns)
print("x.shape",x.shape)
''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
fi_str = "0.06165973 0.00682135 0.03925348 0.04481814 0.14518396 0.24816072\
 0.26320692 0.1908957 "
 
''' str에서 숫자로 변환하는 구간 '''
fi_str = fi_str.split()
fi_float = [float(s) for s in fi_str]
print(fi_float)
fi_list = pd.Series(fi_float)

''' 25퍼 미만 인덱스 구하기 '''
low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
print('low_idx_list',low_idx_list)

''' 25퍼 미만 제거하기 '''
low_col_list = [x.columns[index] for index in low_idx_list]
# 이건 혹여 중복되는 값들이 많아 25퍼이상으로 넘어갈시 25퍼로 자르기
if len(low_col_list) > len(x.columns) * 0.25:   
    low_col_list = low_col_list[:int(len(x.columns)*0.25)]
print('low_col_list',low_col_list)
x.drop(low_col_list,axis=1,inplace=True)
print("after x.shape",x.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.77, shuffle = False, random_state=1266)
print(x_train.shape, x_test.shape) #(7620, 8) (3266, 8)
print(y_train.shape, y_test.shape) #(7620, ) (3266, )


models = [DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]

for model in models:
    model.fit(x_train, y_train)
    results = model.score(x_test,y_test) #분류에서는 (디폴트값)acc 빼줌 회귀는 r2
    y_predict = model.predict(x_test)
    r2 = r2_score(y_predict, y_test)
    print(type(model).__name__, "r2 :", r2)
    print(type(model).__name__, ':', model.feature_importances_)
        



y_submit = model.predict(test_csv)




submission_csv['count'] = y_submit

print(submission_csv)
accuracy_score = ((y_test, y_submit))

y_submit = (y_submit.round(0).astype(int))


#submission_csv.to_csv(path + "submission_29.csv", index= False)
print("음수갯수 :", submission_csv[submission_csv['count']<0].count())
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


# DecisionTreeRegressor r2 : -1.7268433006054646
# DecisionTreeRegressor : [0.06165973 0.00682135 0.03925348 0.04481814 0.14518396 0.24816072
#  0.26320692 0.1908957 ]
# RandomForestRegressor r2 : -4.09371846175132
# RandomForestRegressor : [0.06579598 0.00565688 0.04250754 0.04900778 0.12632627 0.25134967
#  0.26614433 0.19321154]
# GradientBoostingRegressor r2 : -7.128898625035532
# GradientBoostingRegressor : [0.0620696  0.00324965 0.03924756 0.01213083 0.11730789 0.40713346
#  0.34264321 0.01621779]
# XGBRegressor r2 : -4.54582812901873
# XGBRegressor : [0.10264958 0.04851723 0.09922846 0.06810114 0.10283971 0.3736201
#  0.14896719 0.05607657]

#컬럼삭제
# DecisionTreeRegressor r2 : -1.5046401912033422
# DecisionTreeRegressor : [0.05934199 0.04670429 0.14799547 0.26390251 0.27225262 0.20980312]
# RandomForestRegressor r2 : -3.8695629809935372
# RandomForestRegressor : [0.06675677 0.05141086 0.13108817 0.26107532 0.27798339 0.21168551]
# GradientBoostingRegressor r2 : -7.561226320122955
# GradientBoostingRegressor : [0.05932843 0.01220883 0.13692998 0.40767036 0.36349544 0.02036697]
# XGBRegressor r2 : -4.350299053306633
# XGBRegressor : [0.11880647 0.07197957 0.1160459  0.4583395  0.16854551 0.06628299]