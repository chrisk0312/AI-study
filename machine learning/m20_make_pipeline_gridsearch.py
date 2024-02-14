from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
# 대문자로 시작하는 이름은 클래스, 소문자로 시작하는 이름은 함수나 변수
# CamelCase (capitalizing the first letter of each word) for class names 
# and lowercase_with_underscores for function and variable names

#data
datasets = load_iris()
x,y = load_iris(return_X_y=True)
print(x.shape, y.shape) #(150, 4) (150,)
x_train , x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=77,stratify=y)
print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

# scaler =MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
parameters = [
    {"randomforestclassifier__n_estimators":[100,200], "randomforestclassifier__max_depth":[6,10,12],
     "randomforestclassifier__min_samples_leaf":[3,10]},#12
    {"randomforestclassifier__max_depth":[6,8,10,12], "randomforestclassifier__min_samples_leaf":[3,5,7,10]},#16
    {"randomforestclassifier__min_samples_leaf":[3,5,7,10], "randomforestclassifier__min_samples_split":[2,3,5,10]},#16
    {"randomforestclassifier__min_samples_split":[2,3,5,10]},#4
]

# Define a list of models
# model = RandomForestClassifier()

pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier()) #2개의 모델을 연결해주는 파이프라인
#Pipeline is a class that takes a list of tuples. 
#The first element of each tuple is a string that provides the name of the step, 
# and the second element is an instance of the transformer or estimator

#make_pipeline is a function that takes a sequence of transformers and estimators.
#It automatically gives each step a name based on its class.
#The name is just the lowercased class name.

# model = GridSearchCV(pipe, parameters, cv=5, verbose=1)
model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1)
# model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

model.fit(x_train, y_train)    

results = model.score(x_test, y_test)
print('score:', results)
y_pred = model.predict(x_test)
print('y_pred:', y_pred)

accuracy_score = accuracy_score(y_test, y_pred)
print('accuracy_score:', accuracy_score)

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()

clf = GridSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)

# The best parameters are stored in clf.best_params_
print(clf.best_params_)