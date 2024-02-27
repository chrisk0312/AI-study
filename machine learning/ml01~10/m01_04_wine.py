from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

datasets = load_wine()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=77,stratify=y)

#model
model = LinearSVC(C=100)

#compile & fit
model.fit(x_train,y_train)

#evaluate & predict
results = model.score(x_test,y_test)
print("model.score:", results)
y_predict = model.predict(x_test)
print(y_predict)
accuracy_score = accuracy_score(y_predict,y_test)
print("accuracy_score:", accuracy_score)

