import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import time
from sklearn.experimental import enable_halving_search_cv #정식버전이 아님!
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score



#1. 데이터
datasets = fetch_california_housing()
x = np.array(datasets.data)
y = np.array(datasets.target)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle= True, random_state= 123, train_size=0.8)
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)
#n_split = 섞어서 분할하는 갯수
#kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


#2. 모델구성
parameters = [
    {"n_estimators": [500, 1000], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10], "gamma": [0,2,4]},
    {"max_depth": [4, 6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10],"gamma": [0,2,4]},
    {"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10],"gamma": [0,2,4]},
    {"min_samples_split": [2, 3, 5, 10], "subsample": [0, 0.3, 1],"gamma": [0,2,4]},
    {"reg_alpha": [0.2351, 0.3435], "min_samples_split": [3, 5, 10], "max_depth" : [8, 10, 11]}]



model = RandomizedSearchCV(XGBRegressor(),
                              parameters,
                              cv = kfold,
                              verbose = 1,
                               refit= True,
                               random_state= 123,
                               n_jobs= 22)

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()


best_predict = model.best_estimator_.predict(x_test)
best_r2_score = r2_score(y_test, best_predict)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print('best_score :', model.best_score_)
print('score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("r2_score :", r2_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적튠 r2 :", r2_score(y_test, y_predict))

print("걸린시간 :", round(end_time - start_time, 2), "초")

# 최적의 매개변수 :  RandomForestRegressor()
# 최적의 파라미터 :  {'min_samples_split': 2}
# best_score : 0.8047142764400448
# score : 0.6369370382370403
# 걸린시간 : 66.89 초

#randomizer
# 최적의 매개변수 :  RandomForestRegressor(n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'min_samples_split': 2}
# best_score : 0.8024624523380005
# score : 0.6374314954175162
# 걸린시간 : 12.26 초

# 최적의 파라미터 :  {'n_estimators': 500, 'min_samples_leaf': 3, 'max_depth': 10, 'gamma': 0}
# best_score : 0.8161134048753356
# score : 0.591291149142545
# r2_score : 0.591291149142545
# 최적튠 r2 : 0.591291149142545