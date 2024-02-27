#https://dacon.io/competitions/open/235576/mysubmission

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,KFold,cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


#1. 데이터
path = "c:\\_data\\daicon\\ddarung\\"
# print(path + "aaa.csv") 문자그대로 보여줌 c:\_data\daicon\ddarung\aaa.csv
# pandas에서 1차원- Series, 2차원이상은 DataFrame이라고 함.

train_csv = pd.read_csv(path + "train.csv", index_col=0) # \ \\ / // 다 가능( 예약어 사용할때 두개씩 사용) 인덱스컬럼은 0번째 컬럼이다라는뜻.
#print(train_csv)
test_csv = pd.read_csv(path +"test.csv", index_col=0)
#print(test_csv)
submission_csv = pd.read_csv(path + "submission.csv") 
#print(submission_csv)


train_csv = train_csv.fillna(train_csv.mean())  #결측치가 하나라도 있으면 행전체 삭제됨.
test_csv = test_csv.fillna(test_csv.mean())   # (0,mean)
#print(train_csv.isnull().sum())
#print(train_csv.info())
#print(train_csv.shape)      #(1328, 10)
#print(test_csv.info()) # 717 non-null


################# x와 y를 분리 ###########
x = train_csv.drop(['count',], axis=1)
#print(x)
y = train_csv['count']
#print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8, stratify=y)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


print(train_csv.index)


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

    



#로스 : 3175.002197265625
#R2 스코어 : 0.5593716340440571
#RMSE :  56.347159447801296

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus']= False
# plt.figure(figsize= (9,6))
# plt.plot(hist.history['loss'], c = 'red', label = 'loss', marker = '.')
# plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss', marker = '.')
# plt.legend(loc = 'upper right')
# plt.title("따릉이 LOSS")
# plt.xlabel('epoch')
# plt.grid()
# plt.show()




#linear
# model.score : 0.41578303426180474
# R2 스코어 : 0.41578303426180474
# 걸린 시간 : 0.02 초
# RMSE :  60.497446733195545

# acc : [0.79077487 0.76781563 0.73679958 0.78573522 0.78743544] 
#  평균 acc : 0.7737