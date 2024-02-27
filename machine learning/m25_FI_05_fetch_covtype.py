import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier


import time
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import time
import tensorflow as tf
#1. 데이터

start_time = time.time()

datasets = fetch_covtype()

x = datasets.data
y = datasets.target
#print(x.shape, y.shape) #(581012, 54) (581012,)
#print(pd.value_counts(y))
#print(np.unique(y, return_counts= True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510],)

# columns = datasets.feature_names
# columns = x.columns
# x = pd.DataFrame(x,columns=columns)
# print("x.shape",x.shape)
# ''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
# fi_str = "0.03546066 0.03446177 0.43028001 0.49979756"
 
# ''' str에서 숫자로 변환하는 구간 '''
# fi_str = fi_str.split()
# fi_float = [float(s) for s in fi_str]
# print(fi_float)
# fi_list = pd.Series(fi_float)

# ''' 25퍼 미만 인덱스 구하기 '''
# low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
# print('low_idx_list',low_idx_list)

# ''' 25퍼 미만 제거하기 '''
# low_col_list = [x.columns[index] for index in low_idx_list]
# # 이건 혹여 중복되는 값들이 많아 25퍼이상으로 넘어갈시 25퍼로 자르기
# if len(low_col_list) > len(x.columns) * 0.25:   
#     low_col_list = low_col_list[:int(len(x.columns)*0.25)]
# print('low_col_list',low_col_list)
# x.drop(low_col_list,axis=1,inplace=True)
# print("after x.shape",x.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7,  shuffle= True, random_state= 398, stratify= y) #y의 라벨값을 비율대로 잡아줌 #회귀모델에서는 ㄴㄴ 분류에서만 가능
#print(x_train.shape, x_test.shape) #(7620, 8) (3266, 8)
#print(y_train.shape, y_test.shape) #(7620, ) (3266, )
#print(np.unique(y_test, return_counts = True ))


# from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
# from sklearn.preprocessing import StandardScaler, RobustScaler

# #mms = MinMaxScaler()
# mms = StandardScaler()
# #mms = MaxAbsScaler()
# #mms = RobustScaler()

# mms.fit(x_train)
# x_train= mms.transform(x_train)
# x_test= mms.transform(x_test)


#2. 모델구성
models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for model in models:
    model.fit(x_train, y_train)
    results = model.score(x_test,y_test) #분류에서는 (디폴트값)acc 빼줌 회귀는 r2
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_predict, y_test)
    print(type(model).__name__, "acc :", acc)
    print(type(model).__name__, ':', model.feature_importances_)
        

#print("mms = StandardScaler")
#로스 : 0.2562239468097687
#정확도 : 0.8990269899368286

#print('#mms = MaxAbsScaler')
#로스 : 0.2306247502565384
#정확도 : 0.910214364528656

#print('#mms = RobustScaler')
#로스 : 0.23458202183246613
#정확도 : 0.9094054102897644

#minmax
# 로스 : 0.34160080552101135
# 정확도 : 0.8604621887207031

#model.score : 0.7131677987883238

#Linear
# model.score : 0.5243826877180099
# acc : 0.5243826877180099
# 걸린시간 : 348.6212000846863 초


# DecisionTreeClassifier acc : 0.9342814852212227
# DecisionTreeClassifier : [3.37760450e-01 2.73457663e-02 1.65026732e-02 6.26411347e-02
#  4.50715724e-02 1.49051085e-01 3.10543370e-02 3.44946528e-02
#  2.30991747e-02 1.40036190e-01 8.97897907e-03 4.68840286e-03
#  1.20542712e-02 1.54396120e-03 2.32089610e-04 1.01309840e-02
#  2.46987439e-03 1.17812962e-02 1.16818951e-03 6.98214127e-04
#  0.00000000e+00 1.45587812e-04 1.14184133e-04 3.36468048e-03
#  1.37249632e-03 1.13519119e-03 2.89468992e-03 1.32651433e-04
#  6.86703936e-06 1.14374137e-03 1.19639449e-03 0.00000000e+00
#  9.85984448e-04 2.89547220e-03 5.11068505e-04 7.16909090e-03
#  9.39630128e-03 5.24861622e-03 3.03687226e-05 2.70440810e-04
#  9.25290481e-04 6.42009749e-05 6.73199901e-03 2.39864437e-03
#  4.59493677e-03 1.24382263e-02 4.98813190e-03 2.37185561e-04
#  1.07794413e-03 1.03404682e-04 2.27670471e-04 2.44616947e-03
#  3.70832850e-03 1.24074133e-03]
# RandomForestClassifier acc : 0.9522328804846705
# RandomForestClassifier : [2.38772965e-01 4.88106624e-02 3.28511288e-02 6.05220546e-02
#  5.72024684e-02 1.16625118e-01 4.13839926e-02 4.32032826e-02
#  4.15666662e-02 1.09685103e-01 9.99547331e-03 5.43214478e-03
#  1.08351475e-02 3.41046374e-02 1.25752198e-03 1.00639627e-02
#  2.27913439e-03 1.26888523e-02 6.85822726e-04 2.97450707e-03
#  5.84492993e-06 5.42096727e-05 1.06215271e-04 1.23091430e-02
#  2.46092867e-03 9.76795060e-03 4.03796451e-03 3.27663936e-04
#  5.10392726e-06 8.14005746e-04 1.72391927e-03 2.63211421e-04
#  1.00129874e-03 1.94413634e-03 7.52486563e-04 1.56414482e-02
#  1.08180338e-02 4.01309650e-03 1.25321729e-04 4.20912391e-04
#  6.42340461e-04 1.65375806e-04 5.24967140e-03 2.98419143e-03
#  3.86870535e-03 5.41049486e-03 4.27560441e-03 5.74374545e-04
#  1.65441981e-03 7.20751640e-05 6.63958932e-04 1.03322084e-02
#  1.07870293e-02 5.78600853e-03]
# GradientBoostingClassifier acc : 0.7732811639434551
# GradientBoostingClassifier : [6.45152497e-01 6.96684189e-03 1.68382439e-03 3.83660727e-02
#  9.88879108e-03 5.52945226e-02 7.20108212e-03 2.53756946e-02
#  2.61883348e-03 4.28474277e-02 2.56220851e-02 5.90021069e-03
#  1.31905751e-02 1.66809102e-03 3.00143690e-04 1.29442605e-02
#  5.07777492e-03 1.70157349e-02 9.46218923e-04 1.61272983e-03
#  0.00000000e+00 0.00000000e+00 4.98834763e-05 2.33920230e-03
#  1.12684961e-03 6.01480760e-03 1.66879177e-03 2.72937397e-04
#  8.46626926e-06 4.27651472e-04 1.14864437e-03 5.68590458e-05
#  2.78505363e-04 1.59445425e-03 9.69013205e-04 1.68579771e-02
#  1.26043981e-02 1.09889951e-03 0.00000000e+00 6.98926049e-05
#  1.03910259e-03 3.98257993e-05 3.81931389e-03 1.29371420e-03
#  3.55440690e-03 9.34794755e-03 1.26000012e-03 6.67731773e-04
#  1.30153406e-03 5.12023727e-06 8.26547010e-04 3.12118876e-03
#  6.41330924e-03 1.04961200e-03]