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
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler



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

# model = LinearSVC(C=10)
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())


#3. 컴파일, 훈련

model.fit(x_train, y_train)

#4. 평가, 예측

results = model.score(x_test, y_test)
print("model.score: :", results)

y_predict = model.predict(x_test)

result = accuracy_score(y_test, y_predict)

acc = accuracy_score(y_predict, y_test)
print("acc :", acc)

# 로스 : 0.027650095522403717
# 정확도 : 1.0

# linearSVC
# model.score: : 0.9722222222222222
# acc : 0.9722222222222222

#pipeline
# acc : 0.9722222222222222