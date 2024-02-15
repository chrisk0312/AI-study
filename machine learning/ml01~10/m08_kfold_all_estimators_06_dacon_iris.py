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
from sklearn.svm import LinearSVC #softvector machine
from sklearn.linear_model import Perceptron, LogisticRegression , LinearRegression#분류!
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#print(y)

print(x.shape, y.shape) #(120, 4) (120,)
print(np.unique(y, return_counts= True)) #array([0, 1, 2] array([40, 41, 39]

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
#n_split = 섞어서 분할하는 갯수

#2. 모델구성
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

allAlgorithms = all_estimators(type_filter='classifier')
#allAlgorithms = all_estimators(type_filter='regressor')

print('allAlgorithms: ', allAlgorithms) #회귀모델의 갯수 : 55
#print("분류모델의 갯수 :", len(allAlgorithms)) #분류모델의 갯수 : 41
#print("회귀모델의 갯수 :", len(allAlgorithms)) 

for name, algorithm in allAlgorithms:
    try: 
       #2. 모델
        model = algorithm()
        
        
       #3. 훈련
        scores = cross_val_score(model, x_train, y_train, cv=kfold)
        
        print("=================================================================")
        print("================", name, "==============================")
        print("acc :", scores, "\n 평균 acc :", round(np.mean(scores),4))
        #4. 예측
        y_predict = cross_val_predict(model, x_test, y_test, cv= kfold)
        acc= accuracy_score(y_test, y_predict)
        print('cross_val_predict ACC :', acc)
 
   
    except:
        print(name,  '은 에러!')
        #continue

# #3. 훈련
# scores = cross_val_score(model, x_train, y_train, cv=kfold)
# print("acc :", scores, "\n 평균 acc :", round(np.mean(scores),4))

# #4. 예측
# y_predict = cross_val_predict(model, x_test, y_test, cv= kfold)
# print(y_predict)
# print(y_test)

# acc= accuracy_score(y_test, y_predict)
# print('cross_val_predict ACC :', acc)
#로스 : 0.20624160766601562
#acc : 0.9166666666666666


#Linear
#acc : 0.9722222222222222

# acc : [0.91666667 0.875      0.95833333 0.95833333 1.        ] 
#  평균 acc : 0.9417

#Stratified
# acc : [1.         0.95833333 0.91666667 0.79166667 0.95833333] 
#  평균 acc : 0.925