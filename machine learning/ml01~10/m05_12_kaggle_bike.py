import numpy as np
import pandas as pd #판다스에 데이터는 넘파이 형태로 들어가있음.
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score
import time
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyClassifier
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
from sklearn.model_selection import train_test_split,KFold,cross_val_score

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
#n_split = 섞어서 분할하는 갯수

#2. 모델구성
model = DummyClassifier()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)
print("acc :", scores, "\n 평균 acc :", round(np.mean(scores),4))

#2. 모델구성




# LinearSVR
# model.score : -0.5530860077357014
# R2 스코어 : -0.5530860077357014

# acc : [0.01652893 0.01286174 0.01561782 0.01607717 0.01653652]
#  평균 acc : 0.0155