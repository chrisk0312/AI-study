#https://dacon.io/competitions/open/236070/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
import time
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier


#1. 데이터
path = "C:\\_data\\daicon\\iris\\"

train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv") 
print(submission_csv)


print(train_csv.shape) # (120, 5)
print(test_csv.shape) # (30, 4)
print(submission_csv.shape) # (30, 2)

print(train_csv.columns) #'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
#       'petal width (cm)', 'species']

print(train_csv.info())
print(test_csv.info())

x = train_csv.drop(['species'], axis = 1)
#print(x)

y = train_csv['species']

#print(y)

print(x.shape, y.shape) #(120, 4) (120,)
print(np.unique(y, return_counts= True)) #array([0, 1, 2] array([40, 41, 39]





x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.7, shuffle = True, random_state= 123, stratify= y)


#2. 모델구성
models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for model in models:
    model.fit(x_train, y_train)
    results = model.score(x_test,y_test) #분류에서는 (디폴트값)acc 빼줌 회귀는 r2
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_predict, y_test)
    print(type(model).__name__, "acc :", acc)
    print(type(model).__name__, ':', model.feature_importances_)
        

#로스 : 0.20624160766601562
#acc : 0.9166666666666666


#Linear
#acc : 0.9722222222222222


# DecisionTreeClassifier acc : 0.9444444444444444
# DecisionTreeClassifier : [0.         0.03317737 0.09951908 0.86730355]
# RandomForestClassifier acc : 0.9444444444444444
# RandomForestClassifier : [0.09076153 0.03793857 0.44096875 0.43033116]
# GradientBoostingClassifier acc : 0.9444444444444444
# GradientBoostingClassifier : [0.0132889  0.12091413 0.31843555 0.54736141]
# XGBClassifier acc : 0.9444444444444444
# XGBClassifier : [0.00695581 0.13548134 0.56626314 0.29129973]