'''
from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)
# x = np.append(x_train, x_test, axis=0)
x = np.concatenate((x_train, x_test), axis=0)
print(x.shape) #(70000, 28, 28)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

# scaler = StandardScaler()
# x = scaler.fit_transform(x)
print(x.shape) #(70000, 784)

# Perform PCA with n_components set to 0.99
pca = PCA(n_components = x.shape[1])
x = pca.fit_transform(x)

EVR = pca.explained_variance_ratio_

evr_cumsum=np.cumsum(EVR)
print(evr_cumsum)

print(np.argmax(evr_cumsum >= 0.95)+1) #154
print(np.argmax(evr_cumsum >= 0.99)+1) #331
print(np.argmax(evr_cumsum >= 0.999)+1) #486
print(np.argmax(evr_cumsum >= 1.0)+1) #713

# Create a DNN model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(n_components_99,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print('Test Accuracy: %.3f' % accuracy)
'''

from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

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

# Create a DNN model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(n_components_99,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print('Test Accuracy: %.3f' % accuracy)