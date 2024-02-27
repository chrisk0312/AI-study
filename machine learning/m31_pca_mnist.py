'''
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time

# data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(f"{x_train.shape=}\n{x_test.shape=}\n{y_train.shape=}\n{y_test.shape=}")
# x_train.shape=(60000, 28, 28)
# x_test.shape=(10000, 28, 28)
# y_train.shape=(60000,)
# y_test.shape=(10000,)

# x = np.append(x_train,x_test, axis=0)
# x = np.concatenate([x_train,x_test], axis=0)
x = np.vstack([x_train,x_test])
print(x.shape)  # (70000, 28, 28)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=x.shape[1])
x1 = pca.fit_transform(x)
EVR = pca.explained_variance_ratio_
EVR_sum = np.cumsum(EVR)
evr_sum = pd.Series(EVR_sum).round(decimals=4)
print(evr_sum)
print(len(evr_sum[evr_sum >= 0.95]))
print(len(evr_sum[evr_sum >= 0.99]))
print(len(evr_sum[evr_sum >= 0.999]))
print(len(evr_sum[evr_sum >= 1.0]))
print("0.95  커트라인 n_components: ",len(evr_sum[evr_sum < 0.95]))
print("0.99  커트라인 n_components: ",len(evr_sum[evr_sum < 0.99]))
print("0.999 커트라인 n_components: ",len(evr_sum[evr_sum < 0.999]))
print("1.0   커트라인 n_components: ",len(evr_sum[evr_sum < 1.0]))



'''
from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

'''

# Print the number of components
print("Number of components for 99% variance: ", pca.n_components_)



from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the data and labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Concatenate the training and test data
x = np.concatenate((x_train, x_test), axis=0)

# Standardize the data
scaler = StandardScaler()
x = scaler.fit_transform(x.reshape(-1, x.shape[2]*x.shape[1]))

# Perform PCA
pca = PCA(n_components=154)
x = pca.fit_transform(x)

# Concatenate the training and test labels
y = np.concatenate((y_train, y_test), axis=0)

# Split the data and labels into a training set and a test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a RandomForestClassifier object
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(x_train, y_train)

# Print the accuracy of the model on the test set
print('Test Accuracy: %.3f' % clf.score(x_test, y_test))
'''