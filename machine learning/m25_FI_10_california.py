import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

columns = datasets.feature_names
#columns = x.columns
x = pd.DataFrame(x,columns=columns)
print("x.shape",x.shape)
''' 이 밑에 숫자에 얻은 feature_importances 넣고 줄 끝마다 \만 붙여주기'''
fi_str = "0.48506263 0.07050235 0.05032258 0.02355712 0.0219711  0.14934848\
  0.09401683 0.10521888"
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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.15, random_state= 3884 ) #282

print(x)
print(y)
print(x.shape, y.shape)

print(datasets.feature_names)
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)



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
print("r2 :", r2)



#로스 : 0.5067346692085266
#R2 스코어 : 0.6165943551579782
#epochs 5000, batch_size= 200
#8, 16, 10, 8, 4, 1 랜덤 282


#LinearSVR
# model.score : 0.4207089223795233
# r2 : 0.4207089223795233
# 걸린 시간 : 0.39 초

# DecisionTreeRegressor r2 : 0.6566297061118888
# DecisionTreeRegressor : [0.50716354 0.05247775 0.03539131 0.02632487 0.02423904 0.14166997
#  0.10664972 0.10608381]
# RandomForestRegressor r2 : 0.7608411923052989
# RandomForestRegressor : [0.51658824 0.05422539 0.04484719 0.02921317 0.03075699 0.14057888
#  0.09152114 0.09226901]
# GradientBoostingRegressor r2 : 0.7254191469558715
# GradientBoostingRegressor : [0.59743185 0.03362261 0.02091851 0.00450693 0.00192405 0.12865037
#  0.0948059  0.11813978]
# XGBRegressor r2 : 0.8098308025506269
# XGBRegressor : [0.48506263 0.07050235 0.05032258 0.02355712 0.0219711  0.14934848
#  0.09401683 0.10521888]


# 컬럼 삭제
# DecisionTreeRegressor r2 : 0.6540159893196998
# DecisionTreeRegressor : [0.51399049 0.05665913 0.04863564 0.14794639 0.11742469 0.11534366]
# RandomForestRegressor r2 : 0.7599864511521885
# RandomForestRegressor : [0.5231546  0.06011135 0.05690621 0.14960961 0.10541669 0.10480154]
# GradientBoostingRegressor r2 : 0.7276698316496399
# GradientBoostingRegressor : [0.59758972 0.03269591 0.02196864 0.12883044 0.09863082 0.12028447]
# XGBRegressor r2 : 0.815460192474262
# XGBRegressor : [0.51937217 0.07194997 0.05049225 0.15140909 0.10347723 0.10329928]