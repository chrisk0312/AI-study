import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold

#1. data
x,y = load_iris(return_X_y=True)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. modeling
model = SVC()

#3. fit
cross_val_score(model, x, y, cv=kfold)

#4. score
scores = cross_val_score(model, x, y, cv=kfold)
print('ACC:', scores, "\n AVG:", round(np.mean(scores),4))