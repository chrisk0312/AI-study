#https://dacon.io/competitions/open/235610/data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from keras.utils import to_categorical

import time

#1. 데이터
path = "C:\\_data\\daicon\\wine\\"
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)


x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality'] - 3  # quality를 0부터 시작하도록 변환
y = to_categorical(y, num_classes=7)

x['type'] = LabelEncoder().fit_transform(x['type'])
test_csv['type'] = LabelEncoder().fit_transform(test_csv['type'])



x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, train_size=0.8, stratify=y)







# 최적의 매개변수 :  RandomForestClassifier(n_jobs=2)
# 최적의 파라미터 :  {'min_samples_split': 2, 'n_jobs': 2}
# best_score : 0.6502184817457854
# score : 0.5236363636363637
# accuracy_score : 0.5236363636363637
# 최적튠 ACC : 0.5236363636363637
# 걸린시간 : 17.22 초


#randomsearchCV
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'min_samples_split': 3}
# best_score : 0.6506714758506568
# score : 0.52
# accuracy_score : 0.52
# 최적튠 ACC : 0.52
# 걸린시간 : 2.72 초