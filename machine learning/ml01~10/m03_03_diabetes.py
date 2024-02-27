from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
models = [
    LinearSVC(),
    Perceptron(),
    LogisticRegression(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier()
]

# Iterate over models
for model in models:
    # Fit the model
    model.fit(x_train, y_train)
    
    # Evaluate the model
    score = model.score(x_test, y_test)
    print(f'{model.__class__.__name__}: {score}')
