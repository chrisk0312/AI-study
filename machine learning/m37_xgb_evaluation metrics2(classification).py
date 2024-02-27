import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, \
    KFold, StratifiedKFold, \
    GridSearchCV, RandomizedSearchCV, \
    HalvingGridSearchCV, HalvingRandomSearchCV,\
    cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test , y_train, y_test = train_test_split(
    x, y, random_state=777, train_size=0.8,
)

scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=777)
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=777)

# n_estimators : [100,200,300,400,500,1000] 디폴트 100/ 1~inf/ 정수
# learning_rate : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] 디폴트0.3 / 0~1/ eta
# max_depth : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] 디폴트 6 / 0~inf/ 정수
# gamma : [0, 1, 2, 3, 4, 5, 7, 10, 100] 디폴트 0 / 0~inf
# min_child_weight : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100] / 디폴트 1 / 0~inf
# subsample : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# colsample_bytree : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# colsample_bylevel : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# colsample_bynode : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1/ 0~1
# reg_alpha : [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트0 / 0~inf / L1 절대값 가중치 규제
# / alpha
# reg_lambda : [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 1 / 0~inf / L2 제곱 가중치 규제
# / lambda

parameters = {
    'n_estimators' : 100,
    'learning_rate' : 0.1,
    'max_depth' : 6,
    'min_child_weight' : 10
}


model = XGBClassifier()
# model = XGBRegressor()

model.set_params(
    early_stopping_rounds=30,
    **parameters)

model.fit(x_train, y_train,
          eval_set=[(x_train, y_train),(x_test, y_test)],
          verbose=10,
      #   eval_metric='rmse' #regression default
        #   eval_metric='mae' #rmsle,mape,mphe...
        #   eval_metric='logloss'   #binary classification default
        #   eval_metric='mlogloss'   #multi classification default
        #   eval_metric='error'   #binary classification
        #   eval_metric='merror' #multi classification
        #   eval_metric='auc' binary classification, mulit classification
          )

results = model.score(x_test,y_test)
print('model.score :', results)
