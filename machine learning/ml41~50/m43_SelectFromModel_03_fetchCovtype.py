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

x, y = fetch_covtype(return_X_y=True)
y = y-1

import warnings
warnings.filterwarnings('ignore')

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8,
    stratify=y
)

print(x_train.shape,y_train.shape)
print(np.unique(y_train,return_counts=True))


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

model = XGBClassifier()
model.set_params(early_stopping_rounds=10,**xgb_params)
# model = RandomForestClassifier()

# fit & pred
model.fit(x_train,y_train,
          eval_set=[(x_train,y_train), (x_test,y_test)],
          verbose=1,
          eval_metric='mlogloss',
          )

result = model.score(x_test,y_test)
print("Score: ",result)

pred = model.predict(x_test)
acc = accuracy_score(y_test,pred)
print("ACC: ",acc)

# evaluate

# best param :  {'n_estimators': 400, 'min_child_weight': 0.01, 'max_depth': 6, 'learning_rate': 0.1, 'gamma': 0, 'early_stoppint_rounds': 50}
# ACC score  :  0.9527777777777777

'''================================================================'''
thresholds = np.sort(model.feature_importances_)
from sklearn.feature_selection import SelectFromModel

acc_dict = {}
for n, i in enumerate(thresholds):
    selection = SelectFromModel(model,threshold=i,prefit=False)
    print(x_train.shape)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(f"{i}\n{select_x_test.shape=}\n{x_test.shape=}\n=======")
    model2 = XGBClassifier()
    model2.set_params(early_stopping_rounds=10,**xgb_params)
    model2.fit(select_x_train,y_train,
          eval_set=[(select_x_train,y_train), (select_x_test,y_test)],
          verbose=0,
        #   eval_metric='logloss',
          )
    new_result = model2.score(select_x_test,y_test)
    print(f"{n}개 컬럼 삭제, threshold={i:.4f} ACC: {new_result}")
    acc_dict[n] = round(new_result-result,6)
    
for data in acc_dict.items():
    print(data)
    
# (0, 0.0)
# (1, 0.000112)
# (2, 0.000439)
# (3, 0.000749)
# (4, 0.001773)
# (5, 0.000456)
# (6, 0.00136)
# (7, 4.3e-05)
# (8, -2.6e-05)
# (9, -0.000344)
# (10, -0.000482)
# (11, -0.00204)
# (12, 0.000852)
# (13, -0.000129)
# (14, 0.000224)
# (15, 0.000387)
# (16, -0.003786)
# (17, -0.011686)
# (18, -0.01296)
# (19, -0.012797)
# (20, -0.011936)
# (21, -0.012934)
# (22, -0.010895)
# (23, -0.018012)
# (24, -0.016747)
# (25, -0.033545)
# (26, -0.085841)
# (27, -0.142983)
# (28, -0.14349)
# (29, -0.144609)
# (30, -0.145246)
# (31, -0.146606)
# (32, -0.146846)
# (33, -0.147974)
# (34, -0.14843)
# (35, -0.149015)
# (36, -0.150392)
# (37, -0.150857)
# (38, -0.15374)
# (39, -0.154351)
# (40, -0.154635)
# (41, -0.154652)
# (42, -0.154927)
# (43, -0.155289)
# (44, -0.16139)
# (45, -0.166252)
# (46, -0.168085)
# (47, -0.17509)
# (48, -0.182052)
# (49, -0.182224)
# (50, -0.185503)
# (51, -0.193377)
# (52, -0.199831)
# (53, -0.198188