# #change to pandas, then delete column
# #use feature_importances_ to delete 20% of the least important features then reset tha dataset and compare the results with the original model

# from sklearn.datasets import load_iris
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np

# # Load the data
# datasets = load_iris()
# x = datasets.data
# y = datasets.target

# # Split the data
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# # Train a model on the original data
# model = XGBClassifier()
# model.fit(x_train, y_train)

# # Calculate accuracy on the test data
# original_accuracy = accuracy_score(y_test, model.predict(x_test))

# # Calculate feature importances
# importances = model.feature_importances_

# # Find the indices to drop (the 20% least important features)
# num_features_to_drop = int(x.shape[1] * 0.25)
# indices_to_drop = np.argsort(importances)[:num_features_to_drop]

# # Drop the least important features
# x_reduced = np.delete(x, indices_to_drop, axis=1)

# # Split the reduced data
# x_train_reduced, x_test_reduced, y_train, y_test = train_test_split(x_reduced, y, test_size=0.2, random_state=42)

# # Train a model on the reduced data
# model_reduced = XGBClassifier()
# model_reduced.fit(x_train_reduced, y_train)

# # Calculate accuracy on the reduced test data
# reduced_accuracy = accuracy_score(y_test, model_reduced.predict(x_test_reduced))

# # Print the original and reduced accuracies
# print(f"Original accuracy: {original_accuracy}")
# print(f"Reduced accuracy: {reduced_accuracy}")

from sklearn.datasets import load_iris
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Load the data
datasets = load_iris()
x = datasets.data
y = datasets.target

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