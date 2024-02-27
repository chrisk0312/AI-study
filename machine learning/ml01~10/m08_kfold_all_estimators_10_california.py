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

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


n_splits=5
#kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
#n_split = 섞어서 분할하는 갯수
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)


from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score


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
# scores = cross_val_score(model, x, y, cv=kfold)
# print("acc :", scores, "\n 평균 acc :", round(np.mean(scores),4))

# #4. 예측
# y_predict = cross_val_predict(model, x_test, y_test, cv= kfold)
# print(y_predict)
# print(y_test)

# acc= accuracy_score(y_test, y_predict)
# print('cross_val_predict ACC :', acc)


#LinearSVR
# model.score : 0.4207089223795233
# r2 : 0.4207089223795233
# 걸린 시간 : 0.39 초

# acc : [0.82173339 0.8310562  0.81078681 0.79881124 0.80591868] 
#  평균 acc : 0.8137s