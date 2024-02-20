from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier



#1. 데이터

datasets = load_wine()
print(datasets.DESCR) 
print(datasets.feature_names) #'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'       


x= datasets.data
y= datasets.target
print(x.shape, y.shape) #(178, 13) (178,)

print(np.unique(y, return_counts= True)) #(array([0, 1, 2]), array([59, 71, 48]
print(pd.value_counts(y)) # 1    71 # 0    59 # 2    48

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, shuffle= True, random_state= 123, stratify= y)

#2. 모델구성
models = [DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), XGBClassifier()]

for model in models:
    model.fit(x_train, y_train)
    results = model.score(x_test,y_test) #분류에서는 (디폴트값)acc 빼줌 회귀는 r2
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_predict, y_test)
    print(type(model).__name__, "acc :", acc)
    print(type(model).__name__, ':', model.feature_importances_)
        

# 로스 : 0.027650095522403717
# 정확도 : 1.0

# linearSVC
# model.score: : 0.9722222222222222
# acc : 0.9722222222222222


# DecisionTreeClassifier acc : 0.8888888888888888
# DecisionTreeClassifier : [0.         0.         0.02013843 0.02086548 0.         0.
#  0.40558571 0.         0.         0.38697541 0.         0.02097754
#  0.14545744]
# RandomForestClassifier acc : 0.9722222222222222
# RandomForestClassifier : [0.13748499 0.02766604 0.01540948 0.02141288 0.03467347 0.06254837
#  0.16058637 0.00509544 0.0213096  0.14493472 0.0677721  0.1308265
#  0.17028004]
# GradientBoostingClassifier acc : 0.9166666666666666
# GradientBoostingClassifier : [3.96450531e-03 4.27494060e-02 1.69283576e-02 4.35841084e-03
#  0.00000000e+00 5.17074204e-03 2.97038650e-01 2.45026369e-04
#  0.00000000e+00 3.14857050e-01 9.47643680e-03 2.63123523e-02
#  2.78899062e-01]
# XGBClassifier acc : 0.9444444444444444
# XGBClassifier : [0.0633141  0.08874952 0.01123517 0.00261207 0.03230456 0.02632484
#  0.22729877 0.         0.01110675 0.2843193  0.03778306 0.02962446
#  0.18532743]