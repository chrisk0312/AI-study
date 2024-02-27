from sklearn.datasets import load_boston
from keras.models import Sequential
import pandas as pd
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore') #warning 무시. 나타내지않음.
import time
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor


# 현재 사이킷런 버전 1.3.0 보스턴 안됨. 삭제 후 재설치
#pip uninstall scikit-learn
#pip uninstall scikit-image                  
#pip uninstall scikit-learn-intelex
#pip install scikit-learn==1.1.3


datasets = load_boston()
print(datasets)
x = datasets.data
y = datasets.target
print(x)
print(x.shape) # (506, 13)
print(y.shape) # (506,)

print(datasets.feature_names)
#['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
# 'B' 'LSTAT']


columns = datasets.feature_names
#columns = x.columns
x = pd.DataFrame(x,columns=columns)
print("x.shape",x.shape)
''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
fi_str = "2.70772461e-02 3.63384406e-04 1.78974511e-03 3.71753472e-04\
  4.37679491e-02 3.73785755e-01 1.32758585e-02 8.28337336e-02\
  3.98979761e-03 1.02928845e-02 2.89756131e-02 7.81285500e-03\
  4.05663424e-01"
 
''' str에서 숫자로 변환하는 구간 '''
fi_str = fi_str.split()
fi_float = [float(s) for s in fi_str]
print(fi_float)
fi_list = pd.Series(fi_float)

''' 25퍼 미만 인덱스 구하기 '''
low_idx_list = fi_list[fi_list <= fi_list.quantile(0.25)].index
print('low_idx_list',low_idx_list)

''' 25퍼 미만 제거하기 '''
low_col_list = [x.columns[index] for index in low_idx_list]
# 이건 혹여 중복되는 값들이 많아 25퍼이상으로 넘어갈시 25퍼로 자르기
if len(low_col_list) > len(x.columns) * 0.25:   
    low_col_list = low_col_list[:int(len(x.columns)*0.25)]
print('low_col_list',low_col_list)
x.drop(low_col_list,axis=1,inplace=True)
print("after x.shape",x.shape)

print(datasets.DESCR) #Describe #행 = Tnstances

# [실습]
#train_size 0.7이상, 0.9이하
#R2 0.8 이상

#1. 데이터


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.20, random_state= 4041 ) #4041

print(x_train)          
print(y_train)
print(x_test)
print(x_test)


models = [DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), XGBRegressor()]

for model in models:
    model.fit(x_train, y_train)
    results = model.score(x_test,y_test) #분류에서는 (디폴트값)acc 빼줌 회귀는 r2
    y_predict = model.predict(x_test)
    r2 = r2_score(y_predict, y_test)
    print(type(model).__name__, "r2 :", r2)
    print(type(model).__name__, ':', model.feature_importances_)
        



from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 스코어 :", r2)

#로스 : 11.972917556762695
#R2 스코어 : 0.8349828030281251
#test_size= 0.20, random_state= 4041
#epochs = 5000, batch_size= 20
# 노드 - 1, 9, 13, 9, 3, 1

#LinearSVR
# model.score : 0.5005368994698873
# R2 스코어 : 0.5005368994698873
# 걸린 시간 : 0.01 초

# DecisionTreeRegressor r2 : 0.7734854995317868
# DecisionTreeRegressor : [4.15143232e-02 1.45851776e-04 2.78741064e-03 6.54562544e-04
#  3.42472839e-02 5.58714709e-01 2.14211513e-02 7.76494226e-02
#  2.28720045e-03 6.57886646e-03 2.81703113e-02 2.16355114e-02
#  2.04193395e-01]
# RandomForestRegressor r2 : 0.8289373015405717
# RandomForestRegressor : [0.03920761 0.00113788 0.00520905 0.00203367 0.0279558  0.42904374
#  0.01637921 0.06748749 0.00581766 0.01189261 0.01626317 0.01284681
#  0.36472532]
# GradientBoostingRegressor r2 : 0.8385257453904733
# GradientBoostingRegressor : [2.70772461e-02 3.63384406e-04 1.78974511e-03 3.71753472e-04
#  4.37679491e-02 3.73785755e-01 1.32758585e-02 8.28337336e-02
#  3.98979761e-03 1.02928845e-02 2.89756131e-02 7.81285500e-03
#  4.05663424e-01]
# XGBRegressor r2 : 0.8269429330215232
# XGBRegressor : [0.02866496 0.00206649 0.01824216 0.0030793  0.02299185 0.2691067
#  0.01968556 0.08733429 0.01543631 0.02025558 0.03572486 0.00773475
#  0.4696772 ]
# R2 스코어 : 0.8597828614582055

# 컬럼삭제
# DecisionTreeRegressor r2 : 0.6611729328493811
# DecisionTreeRegressor : [0.03932138 0.03346979 0.55784709 0.02097496 0.07653704 0.00231857
#  0.00859797 0.02986281 0.0128067  0.21826368]
# RandomForestRegressor r2 : 0.8410788569036862
# RandomForestRegressor : [0.03593612 0.02403173 0.40397727 0.01488921 0.07533642 0.00522992
#  0.01359344 0.0190518  0.01117475 0.39677934]
# GradientBoostingRegressor r2 : 0.8445052468810755
# GradientBoostingRegressor : [0.02689124 0.04043887 0.37400867 0.01389531 0.08986747 0.00185683
#  0.01220316 0.03102951 0.00622677 0.40358216]
# XGBRegressor r2 : 0.8341695476982575
# XGBRegressor : [0.02645556 0.03502553 0.17639521 0.01516413 0.06535973 0.02000028
#  0.02520528 0.0450307  0.00586101 0.5855026 ]