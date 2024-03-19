from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

#1 데이터
path = "c:\\_data\\dacon\\diabetes\\"
train_csv = pd.read_csv(path + "train.csv",index_col=0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path +"sample_submission.csv",)
print(submission_csv)
print(train_csv.shape) 
print(test_csv.shape) 
print(submission_csv.shape) 
print(train_csv.columns)
print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())

###########x와 y의 값을 분리
x= train_csv.drop(['Outcome'], axis=1) 
print(x)
y = train_csv['Outcome']
print(y)

x_train,x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.3,shuffle=False, random_state=100,)
print(x_train.shape,x_test.shape) #(2177, 8) (8709, 8)
print(y_train.shape, y_test.shape) #(2177,) (8709,)

# Define a list of models
model = make_pipeline(MinMaxScaler(), RandomForestClassifier()) #2개의 모델을 연결해주는 파이프라인

model.fit(x_train, y_train)
    

results = model.score(x_test, y_test)
print('score:', results)
y_pred = model.predict(x_test)
print('y_pred:', y_pred)

accuracy_score = accuracy_score(y_test, y_pred)
print('accuracy_score:', accuracy_score)