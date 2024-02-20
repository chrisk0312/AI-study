# iter_2 이상 돌리기
# min_resourse 조절
# factor 조절

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.experimental import enable_halving_search_cv #정식버전이 아님!
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV
from sklearn.preprocessing import MinMaxScaler
import time

#1. 데이터

x,y = load_iris(return_X_y= True)


x_train, x_test, y_train , y_test = train_test_split(x, y, shuffle= True, random_state=123, train_size=0.8,stratify= y)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {"n_jobs": [-1],"n_estimators": [100, 200], "max_depth": [6, 10, 12], "min_samples_leaf": [3, 10]},
    {"n_jobs": [-1],"max_depth": [6, 8, 10, 12], "min_samples_leaf": [3, 5, 7, 10]},
    {"n_jobs": [-1],"min_samples_leaf": [3, 5, 7, 10], "min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1],"min_samples_split": [2, 3, 5, 10]},
    {"n_jobs": [-1], "min_samples_split": [2, 3, 5, 10]},
]    
     
RF = RandomForestClassifier()
model = HalvingRandomSearchCV(RandomForestClassifier(), 
                     parameters, 
                     cv=kfold, 
                     verbose=1, 
                     min_resources=10,
                     #n_iter=10,
                     refit= True, #디폴트 트루~
                     n_jobs=-1) #CPU 다 쓴다!

start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

from sklearn.metrics import accuracy_score
best_predict = model.best_estimator_.predict(x_test)
best_acc_score = accuracy_score(y_test, best_predict)

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print('best_score :', model.best_score_)
print('score :', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print("accuracy_score :", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적튠 ACC :", accuracy_score(y_test, y_predict))

print("걸린시간 :", round(end_time - start_time, 2), "초")

# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_leaf=3, n_estimators=200)
# 최적의 파라미터 :  {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 200}
# best_score : 0.9583333333333334
# score : 0.9333333333333333
# accuracy_score : 0.9333333333333333
# 최적튠 ACC : 0.9333333333333333
# 걸린시간 : 2.19 초


#randomizer
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=10, min_samples_split=3, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'min_samples_split': 3, 'min_samples_leaf': 10}
# best_score : 0.9583333333333334
# score : 0.9666666666666667
# accuracy_score : 0.9666666666666667
# 최적튠 ACC : 0.9666666666666667
# 걸린시간 : 1.44 초

#harving
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=3, n_jobs=-1)
# 최적의 파라미터 :  {'n_jobs': -1, 'min_samples_split': 3}
# best_score : 0.9666666666666668
# score : 0.9666666666666667
# accuracy_score : 0.9666666666666667
# 최적튠 ACC : 0.9666666666666667
# 걸린시간 : 1.78 초