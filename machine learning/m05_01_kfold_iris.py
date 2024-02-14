import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold

#1. data
x,y = load_iris(return_X_y=True)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

# K-Fold Cross Validation, often referred to as KFold, 
# is a method used in machine learning to estimate 
# the skill of a machine learning model on unseen data. 
# It's a resampling procedure used to evaluate machine learning models
# on a limited data sample.
# The procedure has a single parameter called k 
# that refers to the number of groups that a given data sample 
# is to be split into. 
# For example, if you provide k=5, it means the data will be 
# split into 5 groups or folds.
# The process is repeated k times and for each time, 
# one group is selected as the validation set 
# and the remaining groups are used as the training set.
# A model is trained on the training set and 
# evaluated on the validation set.
# The result is often given as the mean of the model skill scores 
# and provides a more robust estimate of the skill of the model 
# on new data.

#2. modeling
model = SVC()

#3. fit
cross_val_score(model, x, y, cv=kfold)

#4. score
scores = cross_val_score(model, x, y, cv=kfold)
print('ACC:', scores, "\n AVG:", round(np.mean(scores),4))