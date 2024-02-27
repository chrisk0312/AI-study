#https://dacon.io/competitions/open/236068/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
import time


#1. 데이터

x,y = load_diabetes(return_X_y=True)



from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler




x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=777, shuffle= True)#, stratify=y)



# scaler = MinMaxScaler()
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

split = 3
#kfold = StratifiedKFold(n_splits=split, shuffle= True, random_state=123)
kfold = KFold(n_splits=split, shuffle= True, random_state=123)


#2. 모델구성
    

#xgb파라미터
# 'n_estimator : 디폴트 100/ 1~inf/ 정수
'''learning_rate : 디폴트 0.3/ 0~1 /eta''' #적을수록 좋다
#max_depth 디폴트 6 / 0~inf / 정수 #트리의 깊이
# 'gamma' : 디폴트 0/ 0~inf #값이 클수록 데이터 손실이 생김 작을수록 많이 분할되지만 트리가 복잡해짐.
# 'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100]/ 디폴트1/ 0~inf
# 'subsample' : [0, 0.1, 0.3~]/ 디폴트1/ 0~1
# 'colsample_bytree' : [0, 0.1, 0.2, 0.3~~]/ 디폴트1/ 0~1
# 'colsample_bylevel' : [0, 0.1, 0.5]/ 디폴트1 / 0~1
# 'colsample_bynode' : [0, 0.1, 0.5]/ 디폴트1 / 0~1
# 'reg_alpha': [0, 0.1, 0.001]/ 디폴트 0 / 0 ~ inf/ L1 절대값 가중치 규제/ alpha #음수쪽 규제
# 'reg_lambda': [0, 0.1, 0.001]/ 디폴트 1 / 0 ~ inf/ L2 제곱 가중치 규제/ lambda 

parameters = [
    {"n_estimators": [500, 800], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10], "gamma": [0,2,1]},
    {"max_depth": [4, 6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10],"gamma": [0,2,1]},
    {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10],"gamma": [0,2,1]},
    {"min_samples_split": [2, 3, 5, 10], "subsample": [0, 0.3, 1],"gamma": [0,2,4]},
    {"reg_alpha": [0.2351, 0.135], "min_samples_split": [2, 3, 5], "max_depth" : [8, 10, 11]}]



model = RandomizedSearchCV(XGBRegressor(),
                              parameters,
                              cv = kfold,
                              verbose = 1,
                               refit= True,
                               random_state= 123,
                               n_jobs= 22)

#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


#4. 평가, 예측
best_predict = model.best_estimator_.predict(x_test)

from sklearn.metrics import accuracy_score
best_predict = model.best_estimator_.predict(x_test)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print('best_score :', model.best_score_)
print('score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)

y_pred_best = model.best_estimator_.predict(x_test)

print("걸린시간 :", round(end_time - start_time, 2), "초")