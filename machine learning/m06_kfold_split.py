from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold

#data
datasets = load_iris()

df =pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df)

n_splits = 3
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

# KFold object is created to split a dataset into 3 folds. 
# The split method is then used to generate the indices for the training and validation sets for each fold.
# The difference between KFold and KFold.split is that KFold is a class that provides the functionality to perform K-Fold Cross Validation,
# while KFold.split is a specific method of the KFold class that generates the train/test indices.


for train_index, val_index in kfold.split(df):
    print("================================")
    print(train_index, "\n", val_index)
    print(len(train_index), len(val_index))
