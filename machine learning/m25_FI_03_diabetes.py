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



# Convert to DataFrame
df = pd.DataFrame(x, columns=datasets.feature_names)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

# Train a model on the original data
model = XGBClassifier()
model.fit(x_train, y_train)

# Calculate accuracy on the test data
original_accuracy = accuracy_score(y_test, model.predict(x_test))

# Calculate feature importances
importances = model.feature_importances_

# Find the features to drop (the 20% least important features)
num_features_to_drop = int(df.shape[1] * 0.25)
features_to_drop = df.columns[np.argsort(importances)[:num_features_to_drop]]

# Drop the least important features
df_reduced = df.drop(features_to_drop, axis=1)

# Split the reduced data
x_train_reduced, x_test_reduced, y_train, y_test = train_test_split(df_reduced, y, test_size=0.2, random_state=42)

# Train a model on the reduced data
model_reduced = XGBClassifier()
model_reduced.fit(x_train_reduced, y_train)

# Calculate accuracy on the reduced test data
reduced_accuracy = accuracy_score(y_test, model_reduced.predict(x_test_reduced))

# Print the original and reduced accuracies
print(f"Original accuracy: {original_accuracy}")
print(f"Reduced accuracy: {reduced_accuracy}")