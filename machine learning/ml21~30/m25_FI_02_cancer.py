from sklearn.datasets import load_breast_cancer
from sklearn.svm import LinearSVC
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
#1.데이터
datasets = load_breast_cancer()
x = datasets.data
y= datasets.target


# Convert to DataFrame
df = pd.DataFrame(x, columns=datasets.feature_names)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

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