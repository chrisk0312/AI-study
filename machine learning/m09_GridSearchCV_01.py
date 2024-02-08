import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
import time
import pandas as pd
#1. data
x,y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=123, stratify=y)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {"C":[1,10,100,1000], "kernel":["linear"], "degree":[3,4,5]},#12
    {"C":[1,10,100,], "kernel":["rbf"], "gamma":[0.001, 0.0001]},#6
    {"C":[1,10,100,1000], "kernel":["sigmoid"],#24
    "gamma":[0.1,0.001, 0.0001], "degree":[3,4]},
]

#2. modeling
# model = SVC(C=1, kernel='linear',degree=3)
model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1,
                     refit=True, n_jobs=-1)
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() 
print('최적의 매개변수:', model.best_estimator_)#최적의 매개변수: SVC(C=1, kernel='linear')
print('최적의 매개변수:', model.best_params_)#최적의 매개변수: {'C': 1, 'gamma': 0.1, 'kernel': 'sigmoid'}
print('best_score:', model.best_score_)# best_score: 0.975
print('model.score:', model.score(x_test, y_test))#model.score: 0.9666666666666667

y_predict = model.predict(x_test)
print("acc:", accuracy_score(y_predict,y_test))#acc: 0.9666666666666667

y_predict_best = model.best_estimator_.predict(x_test) 
print("acc:", accuracy_score(y_predict_best,y_test))#acc: 0.9666666666666667
print("걸린시간:", round(end_time- start_time,2)) #걸린시간: 1.5781776905059814
print(pd.DataFrame(model.cv_results_).T)

