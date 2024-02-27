#https://dacon.io/competitions/open/235610/data

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
import time
from sklearn.svm import SVR 
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
from sklearn.model_selection import train_test_split,KFold,cross_val_score


#1. 데이터
path = "C:\\_data\\daicon\\wine\\"



train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
print(train_csv)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv)

print(train_csv.shape) #(5497, 13)
print(test_csv.shape) #(1000, 12)
print(submission_csv.shape) #(1000, 2)


print(train_csv.columns) #'quality', 'fixed acidity', 'volatile acidity', 'citric acid',
    #    'residual sugar', 'chlorides', 'free sulfur dioxide',
    #    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
    #    'type'],
    
x = train_csv.drop(['quality'], axis= 1)
print(x)
y = train_csv['quality']



x.loc[x['type'] == 'red', 'type'] = 1
x.loc[x['type'] == 'white', 'type'] = 0
print(x)

test_csv.loc[test_csv['type'] == 'red', 'type'] = 1
test_csv.loc[test_csv['type'] == 'white', 'type'] = 0

print(test_csv)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
#n_split = 섞어서 분할하는 갯수

#2. 모델구성
model = ExtraTreesClassifier()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)
print("acc :", scores, "\n 평균 acc :", round(np.mean(scores),4))



#LinearSVR()
# model.score: 0.2888460317435524
# 로스 : 0.2888460317435524

# acc : [0.65454545 0.68181818 0.66424022 0.67788899 0.67424932] 
#  평균 acc : 0.6705