#r2 0.55 ~0.6이상
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.datasets import fetch_california_housing
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np

#1.데이터
datasets = fetch_california_housing()
x =datasets.data
y =datasets.target


n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. modeling
model = SVC()

#3. fit
cross_val_score(model, x, y, cv=kfold)

#4. score
scores = cross_val_score(model, x, y, cv=kfold)
print('ACC:', scores, "\n AVG:", round(np.mean(scores),4))