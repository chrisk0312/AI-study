import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score

#1 데이터

path = "c:\_data\dacon\iris\\"
train_csv = pd.read_csv(path + "train.csv", index_col =0)
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv")
print(submission_csv)
print(submission_csv)
print(train_csv.shape) 
print(test_csv.shape) 
print(submission_csv.shape) 
print(train_csv.columns)
print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())

x= train_csv.drop(['species'], axis=1) 
print(x)
y = train_csv['species']
print(y)


n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. modeling
model = SVC()

#3. fit
cross_val_score(model, x, y, cv=kfold)

#4. score
scores = cross_val_score(model, x, y, cv=kfold)
print('ACC:', scores, "\n AVG:", round(np.mean(scores),4))