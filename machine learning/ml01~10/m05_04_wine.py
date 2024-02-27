from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score

datasets = load_wine()
x = datasets.data
y = datasets.target


n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. modeling
model = SVC()

#3. fit
cross_val_score(model, x, y, cv=kfold)

#4. score
scores = cross_val_score(model, x, y, cv=kfold)
print('ACC:', scores, "\n AVG:", round(np.mean(scores),4))
