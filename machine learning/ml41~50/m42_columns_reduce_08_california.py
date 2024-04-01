from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import fetch_covtype
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

import warnings
warnings.filterwarnings(action='ignore')
#data

x, y = fetch_california_housing(return_X_y=True)

# print(np.unique(y,return_counts=True)) # 회귀 
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

y = LabelEncoder().fit_transform(y)
print(x.shape,y.shape)
print(np.unique(y,return_counts=True))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8,
    # stratify=y
)



sclaer = MinMaxScaler().fit(x_train)
x_train = sclaer.transform(x_train)
x_test = sclaer.transform(x_test)


xgb_params = {'learning_rate': 0.13349839953884737,
                'n_estimators': 99,
                'max_depth': 8,
                'min_child_weight': 3.471164143831403e-06,
                'subsample': 0.6661302167437514,            #dropout 비슷
                'colsample_bytree': 0.9856906281904222,
                'gamma': 4.5485144879936555e-06,
                'reg_alpha': 0.014276113125688179,
                'reg_lambda': 10.121476098960851,
                # 'nthread' : 20,
                'tree_method' : 'gpu_hist',
                'predictor' : 'gpu_predictor',
                }

model = XGBRFRegressor()
model.set_params(early_stopping_rounds=10,**xgb_params)
# model = RandomForestClassifier()

# fit & pred
model.fit(x_train,y_train,
          eval_set=[(x_train,y_train), (x_test,y_test)],
          verbose=1,
          eval_metric='rmse',
          )

result = model.score(x_test,y_test)
print("Score: ",result)

pred = model.predict(x_test)
# acc = accuracy_score(y_test,pred)
# print("ACC: ",acc)

# evaluate

# best param :  {'n_estimators': 400, 'min_child_weight': 0.01, 'max_depth': 6, 'learning_rate': 0.1, 'gamma': 0, 'early_stoppint_rounds': 50}
# ACC score  :  0.9527777777777777

'''================================================================'''
result = model.score(x_test,y_test)
feature_importances_list = list(model.feature_importances_)
feature_importances_list_sorted = sorted(feature_importances_list)
# print(feature_importances_list)
drop_feature_idx_list = [feature_importances_list.index(feature) for feature in feature_importances_list_sorted] # 중요도가 낮은 column인덱스 부터 기재한 리스트
print(drop_feature_idx_list)

result_dict = {}
for i in range(len(drop_feature_idx_list)-1): # 1바퀴에는 1개, 마지막 바퀴에는 29개 지우기, len -1해준 이유는 30개 지우면 안되니까
    drop_idx = drop_feature_idx_list[:i+1] # +1 해준 이유는 첫바퀴에 0개가 아니라 1개를 지워야하니까
    new_x_train = np.delete(x_train,drop_idx,axis=1)
    new_x_test = np.delete(x_test,drop_idx,axis=1)
    print(new_x_train.shape,new_x_test.shape)
    
    model.fit(new_x_train,y_train,
          eval_set=[(new_x_train,y_train), (new_x_test,y_test)],
          verbose=0,
          eval_metric='rmse',
          )
    new_result = model.score(new_x_test,y_test)
    print(f"{i+1}개 컬럼이 삭제되었을 때 Score: ",new_result)
    result_dict[i+1] = new_result - result    # 그대로 보면 숫자가 비슷해서 구분하기 힘들기에 얼마나 변했는지 체크
    
for data in result_dict.items():
    print(data)
    
# (1, -0.0005155330227871735)
# (2, 0.0007182862867110007)
# (3, -0.0017647623777153898)
# (4, -0.022911266689605303)
# (5, -0.05115847511218452)
# (6, -0.09379806761858844)
# (7, -0.04357793339310012)