import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time
from sklearn.svm import LinearSVR
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
datasets = fetch_california_housing()
x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


n_splits=5
#kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
#n_split = 섞어서 분할하는 갯수
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)


#2. 모델구성
model = ExtraTreesRegressor()

#3. 훈련
scores = cross_val_score(model, x, y, cv=kfold)
print("acc :", scores, "\n 평균 acc :", round(np.mean(scores),4))

#4. 예측
y_predict = cross_val_predict(model, x_test, y_test, cv= kfold)
print(y_predict)
print(y_test)




#LinearSVR
# model.score : 0.4207089223795233
# r2 : 0.4207089223795233
# 걸린 시간 : 0.39 초

# acc : [0.82173339 0.8310562  0.81078681 0.79881124 0.80591868] 
#  평균 acc : 0.8137