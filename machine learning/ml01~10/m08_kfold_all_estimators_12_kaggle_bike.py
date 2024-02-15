import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
import time
from sklearn.svm import LinearSVC #softvector machine
from sklearn.linear_model import Perceptron, LogisticRegression , LinearRegression#분류!
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils import all_estimators
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1.데이터
datasets = load_breast_cancer()


x = datasets.data
y= datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


n_splits=5
#kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
#n_split = 섞어서 분할하는 갯수
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

n_splits=5


#2. 모델구성
model = SVC()

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

#mms = MinMaxScaler()
#mms = StandardScaler()
#mms = MaxAbsScaler()
mms = RobustScaler()


#2. 모델구성
model = SVC()




#3. 컴파일, 훈련
scores = cross_val_score(model, x, y, cv=kfold)
print("acc :", scores, "\n 평균 acc :", round(np.mean(scores),4))

#4. 예측
y_predict = cross_val_predict(model, x_test, y_test, cv= kfold)
print(y_predict)
print(y_test)

acc= accuracy_score(y_test, y_predict)
print('cross_val_predict ACC :', acc)


# acc : [0.92982456 0.92105263 0.92982456 0.92982456 0.87610619] 
#  평균 acc : 0.9173

#stratifiedKFold
# acc : [0.88596491 0.92982456 0.94736842 0.92105263 0.91150442] 
#  평균 acc : 0.9191
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
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)



n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
#n_split = 섞어서 분할하는 갯수

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





# LinearSVR
# model.score : -0.5530860077357014
# R2 스코어 : -0.5530860077357014

# acc : [0.01652893 0.01286174 0.01561782 0.01607717 0.01653652]
#  평균 acc : 0.0155


#LinearSVC
# model.sore: 0.9617486338797814
# acc : 0.9617486338797814

#Scaler
# acc : [0.88596491 0.92982456 0.94736842 0.92105263 0.91150442] 
#  평균 acc : 0.9191