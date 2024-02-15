from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the data and labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Concatenate the training and test data
x = np.concatenate((x_train, x_test), axis=0)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

# Perform PCA with n_components set to the total number of features
pca = PCA(n_components = x.shape[1])
x = pca.fit_transform(x)

# Calculate the cumulative sum of explained variance ratio
EVR = pca.explained_variance_ratio_
evr_cumsum = np.cumsum(EVR)

# Find the number of components for 99% variance
n_components_99 = np.argmax(evr_cumsum >= 0.99) + 1

# Perform PCA again with n_components set to the number of components for 99% variance
pca = PCA(n_components=n_components_99)
x = pca.fit_transform(x)

# Concatenate the training and test labels
y = np.concatenate((y_train, y_test), axis=0)

# Split the data and labels into a training set and a test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create an XGBClassifier model
model = XGBClassifier(use_label_encoder=False)

# Train the model
model.fit(x_train, y_train, eval_metric='mlogloss')

# Make predictions on the test set
y_pred = model.predict(x_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Test Accuracy: %.3f' % accuracy)

