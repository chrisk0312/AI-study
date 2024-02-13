from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

#1.데이터
datasets = load_breast_cancer()
x,y = load_breast_cancer(return_X_y=True)
print(x.shape, y.shape) #(150, 4) (150,)
x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=77,stratify=y)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

print(np.unique(y, return_counts= True)) #(array([0, 1]), array([212, 357])
print(pd.Series(y).value_counts()) #다 똑가틈
print(pd.DataFrame(y).value_counts()) #다 똑가틈 (dataframe을 쓰면 메모리를 더 씀,행렬데이터는 dataframe만 가능)
print(pd.value_counts(y)) #다 똑가틈
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.68, shuffle= False, random_state= 334)

model = make_pipeline(MinMaxScaler(), RandomForestClassifier()) #2개의 모델을 연결해주는 파이프라인

model.fit(x_train, y_train)
    

results = model.score(x_test, y_test)
print('score:', results)
y_pred = model.predict(x_test)
print('y_pred:', y_pred)

accuracy_score = accuracy_score(y_test, y_pred)
print('accuracy_score:', accuracy_score)
