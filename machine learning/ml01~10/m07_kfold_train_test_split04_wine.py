from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC #softvector machine
from sklearn.linear_model import Perceptron, LogisticRegression , LinearRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터

datasets = load_wine()
print(datasets.DESCR) 
print(datasets.feature_names) #'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'       


x= datasets.data
y= datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
#n_split = 섞어서 분할하는 갯수

#2. 모델구성
model = RidgeClassifier()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print("acc :", scores, "\n 평균 acc :", round(np.mean(scores),4))

#4. 예측
y_predict = cross_val_predict(model, x_test, y_test, cv= kfold)
print(y_predict)
print(y_test)

acc= accuracy_score(y_test, y_predict)
print('cross_val_predict ACC :', acc)


# linearSVC
# model.score: : 0.9722222222222222
# acc : 0.9722222222222222

# acc : [1.         1.         1.         1.         0.94285714] 
#  평균 acc : 0.9886

#Stratified
# acc : [1.         1.         0.97222222 1.         0.97142857] 
#  평균 acc : 0.9887

# cross_val_predict ACC : 0.9444444444444444