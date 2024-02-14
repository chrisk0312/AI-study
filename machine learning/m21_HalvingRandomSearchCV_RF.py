import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import pandas as pd
import sklearn as sk
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

#1. data
x,y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=123, stratify=y)

print(x_train.shape, x_test.shape) #(1437, 64) (360, 64
n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# parameters = [
#     {"C":[1,10,100,1000], "kernel":["linear"], "degree":[3,4,5]},#12
#     {"C":[1,10,100,], "kernel":["rbf"], "gamma":[0.001, 0.0001]},#6
#     {"C":[1,10,100,1000], "kernel":["sigmoid"],#24
#     "gamma":[0.1,0.001, 0.0001], "degree":[3,4]},
# ]

param_distributions = {
    'C': [1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'degree': [3, 4, 5],
    'gamma': [0.1, 0.001, 0.0001]
}

# C: Regularization parameter. 
# The strength of the regularization is inversely proportional to C. 
# Must be strictly positive.

# kernel: Specifies the kernel type to be used in the algorithm.

# degree: Degree of the polynomial kernel function ('poly'). 
# Ignored by all other kernels.

# gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

#2. modeling
# model = SVC(C=1, kernel='linear',degree=3)
# model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1,
#                      refit=True, n_jobs=-1)
# model = RandomizedSearchCV(SVC(), parameters, cv=kfold, verbose=1,
#                            refit=True, n_jobs=-1, random_state=66, 
#                            n_iter=10, scoring='accuracy')
#default = 10, cv=5 = 50 fits 

# model = HalvingGridSearchCV(SVC(), parameters, 
model = HalvingRandomSearchCV(SVC(), param_distributions, 
                           cv=kfold,
                           verbose=1,
                           refit=True, n_jobs=-1, random_state=66, 
                           factor=3,
                           min_resources=150,
                        #  n_iter=10,
        )

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

print(sk.__version__)
